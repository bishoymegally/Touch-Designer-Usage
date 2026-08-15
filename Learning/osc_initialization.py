from pythonosc.udp_client import SimpleUDPClient
def initialize(get_port):
    ip = "127.0.0.1"
    port = get_port

    client = SimpleUDPClient(ip, port)
    return client
