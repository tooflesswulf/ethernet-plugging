import socket
import atexit
import threading

from promise import Promise

'''
Remaining TODO:
- impl & handle AUTOSEND()
'''


class ActionResult:
    """Result of an action command sent to the WSG gripper.
    (MOVE, GRIP, HOME, RELEASE):
        .ack     - Promise that resolves on ACK
        .finished - Promise that resolves on FIN
    """

    def __init__(self, ack: Promise, finished: Promise = None):
        self.ack = ack
        self.finished = finished if finished is not None else ack

    def wait(self, timeout=None):
        return self.finished.wait(timeout)

class QueryResult:
    """Result of a query command sent to the WSG gripper.
    (FORCE?, POS?):
        .promise - Promise that resolves with the query result
    """

    def __init__(self, promise: Promise):
        self.promise = promise
        self.promise.then(self._set_value)
        self._val = None

    def _set_value(self, val):
        self._val = val

    def wait(self, timeout=None):
        return self.promise.wait(timeout)

    @property
    def value(self):
        if not self.promise._event.is_set():
            self.wait()
        return self._val


class WSG:
    def __init__(self, ip='192.168.1.20', port=1000):
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((ip, port))
        self._pending_action = None  # (key, {"ack": Promise, "fin": Promise})
        self._pending_queries = {}   # key -> {"result": Promise}
        self._pending_stop = None    # Promise (resolves on ACK STOP)
        self._lock = threading.Lock()
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

        self.ack_fast_stop().finished.wait()
        atexit.register(self.bye)

    def _read_loop(self):
        buf = b''
        while self._running:
            try:
                data = self.tcp_sock.recv(1024)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                self._dispatch(line.decode('utf-8').strip())

    def _clear_pending_action(self, _=None):
        with self._lock:
            self._pending_action = None

    def _dispatch(self, line: str):
        # ACK <CMD>
        if line.startswith('ACK '):
            cmd = line[4:].strip()
            if cmd == 'STOP':
                stop_ack = None
                with self._lock:
                    stop_ack = self._pending_stop
                    self._pending_stop = None
                if stop_ack:
                    stop_ack.resolve()
            else:
                entry = None
                with self._lock:
                    if self._pending_action and self._pending_action[0] == cmd:
                        entry = self._pending_action[1]
                if entry:
                    entry['ack'].resolve()

        # FIN <CMD>
        elif line.startswith('FIN '):
            entry = None
            with self._lock:
                if self._pending_action and self._pending_action[0] == line[4:].strip():
                    entry = self._pending_action[1]
            if entry:
                entry['fin'].resolve()

        # ERR <CMD> / ERR <detail>
        elif line.startswith('ERR'):
            err = RuntimeError(f'WSG error: {line}')
            with self._lock:
                action_entry = self._pending_action[1] if self._pending_action else None
            if action_entry:
                action_entry['ack'].reject(err)
                if 'fin' in action_entry:
                    action_entry['fin'].reject(err)

        # <KEY>=<VALUE> (query responses like FORCE=12.5)
        elif '=' in line:
            key, _, val = line.partition('=')
            key = key.strip()
            with self._lock:
                entry = self._pending_queries.pop(key, None)
            if entry is None:
                return
            try:
                parsed = float(val)
            except ValueError:
                parsed = val.strip()
            entry['result'].resolve(parsed)

    def send(self, msg):
        if isinstance(msg, str):
            msg = msg.encode()
        text = msg.decode().strip()
        if not msg.endswith(b'\n'):
            msg += b'\n'

        is_query = text.endswith('?')
        is_setter = not is_query and '=' in text

        # Derive the key the server will reference in its response.
        # "MOVE(50)\n" -> cmd_name "MOVE"
        # "FORCE?\n"   -> query_key "FORCE"
        # "PWT=5.0\n"  -> setter_key "PWT"
        if is_query:
            key = text.rstrip('?')
        elif is_setter:
            key = text.partition('=')[0]
        else:
            paren = text.find('(')
            key = text[:paren] if paren != -1 else text

        with self._lock:
            if is_query or is_setter:
                if key in self._pending_queries:
                    raise RuntimeError(f'Query already pending: {key}')
                p = Promise()
                self._pending_queries[key] = {'result': p}
                out = QueryResult(p)
            else:
                if self._pending_action is not None:
                    raise RuntimeError(f'Action already pending: {self._pending_action[0]}')
                if key in ('HOME', 'MOVE', 'GRIP', 'RELEASE'):
                    # Motion commands have separate ACK and FIN phases
                    ack = Promise()
                    fin = Promise()
                else:
                    ack = Promise()
                    fin = ack
                fin.then(self._clear_pending_action).catch(self._clear_pending_action)
                self._pending_action = (key, {'ack': ack, 'fin': fin})
                out = ActionResult(ack, fin)
            self.tcp_sock.sendall(msg)

        return out

    def home(self):
        return self.send(b'HOME()\n')

    def move(self, position, speed=None):
        if speed is not None:
            return self.send(f'MOVE({position}, {speed})\n'.encode())
        return self.send(f'MOVE({position})\n'.encode())

    def grip(self, force=None, width=None, speed=None):
        if force is None:
            return self.send(b'GRIP()\n')
        if width is None:
            return self.send(f'GRIP({force})\n'.encode())
        if speed is None:
            return self.send(f'GRIP({force}, {width})\n'.encode())
        return self.send(f'GRIP({force}, {width}, {speed})\n'.encode())

    def release(self, pullback=10, speed=None):
        if pullback is None:
            return self.send(b'RELEASE()\n')
        if speed is None:
            return self.send(f'RELEASE({pullback})\n'.encode())
        return self.send(f'RELEASE({pullback}, {speed})\n'.encode())

    def force(self):
        return self.send(b'FORCE?\n')

    def position(self):
        return self.send(b'POS?\n')

    def set_pwt(self, value):
        return self.send(f'PWT={value}\n'.encode())

    def set_clt(self, value):
        return self.send(f'CLT={value}\n'.encode())

    def ack_fast_stop(self):
        return self.send(b'FSACK()\n')

    def stop(self):
        ack = Promise()
        with self._lock:
            self._pending_stop = ack
            self.tcp_sock.sendall(b'STOP()\n')
        return CommandResult(ack)

    def bye(self):
        self._running = False
        try:
            self.tcp_sock.sendall(b'BYE()\n')
            self.tcp_sock.close()
        except OSError:
            pass
