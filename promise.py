import threading


class Promise:
    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._error = None
        self._callbacks = []
        self._errbacks = []
        self._lock = threading.Lock()

    def resolve(self, value=None):
        with self._lock:
            self._value = value
            self._event.set()
            callbacks = list(self._callbacks)
        for cb in callbacks:
            cb(value)

    def reject(self, error):
        with self._lock:
            self._error = error
            self._event.set()
            errbacks = list(self._errbacks)
        for eb in errbacks:
            eb(error)

    def then(self, callback):
        next_promise = Promise()

        def handle(value):
            try:
                result = callback(value)
                if isinstance(result, Promise):
                    result.then(next_promise.resolve).catch(next_promise.reject)
                else:
                    next_promise.resolve(result)
            except Exception as e:
                next_promise.reject(e)

        with self._lock:
            if self._event.is_set() and self._error is None:
                handle(self._value)
            elif self._event.is_set():
                next_promise.reject(self._error)
            else:
                self._callbacks.append(handle)
                self._errbacks.append(next_promise.reject)
        return next_promise

    def catch(self, errback):
        next_promise = Promise()

        def handle(error):
            try:
                result = errback(error)
                if isinstance(result, Promise):
                    result.then(next_promise.resolve).catch(next_promise.reject)
                else:
                    next_promise.resolve(result)
            except Exception as e:
                next_promise.reject(e)

        with self._lock:
            if self._event.is_set() and self._error is not None:
                handle(self._error)
            elif self._event.is_set():
                next_promise.resolve(self._value)
            else:
                self._errbacks.append(handle)
                self._callbacks.append(next_promise.resolve)
        return next_promise

    def wait(self, timeout=None):
        self._event.wait(timeout)
        if not self._event.is_set():
            raise TimeoutError()
        if self._error is not None:
            raise self._error
        return self._value
