def test_main_importable():
    import main  # noqa: F401


def test_routes_package_importable():
    from backend.app import routes  # noqa: F401


def test_ai_route_module_importable():
    from backend.app.routes import ai  # noqa: F401
