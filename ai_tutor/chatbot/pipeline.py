from social_core.exceptions import AuthForbidden

def custom_auth_forbidden(backend, details, user=None, is_new=False, *args, **kwargs):
    raise AuthForbidden(
        backend,
        details,
        user=user,
        message='Access denied. Your credentials are not allowed.'
    )
