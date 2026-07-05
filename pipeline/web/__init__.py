"""Local stdlib web server for RUSH run lifecycle APIs."""

__all__ = ["SERVER_VERSION", "RushWebRequestHandler", "create_server"]


def __getattr__(name: str):
    if name in __all__:
        from .server import SERVER_VERSION, RushWebRequestHandler, create_server

        values = {
            "SERVER_VERSION": SERVER_VERSION,
            "RushWebRequestHandler": RushWebRequestHandler,
            "create_server": create_server,
        }
        return values[name]
    raise AttributeError(name)
