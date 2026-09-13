from sqlalchemy.exc import SQLAlchemyError

from .database import verify_connection


def main() -> int:
    try:
        verify_connection()
    except SQLAlchemyError as error:
        print(f"Database connection failed: {error}")
        return 1

    print("Database connection successful")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
