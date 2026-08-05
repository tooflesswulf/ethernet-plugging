import psutil

def is_network_up(iface='enx7cc2c6453f68'):
    stats = psutil.net_if_stats()
    return stats[iface].isup
