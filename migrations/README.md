# Database Migrations

Managed by [Flyway](https://flywaydb.org/).

## Flyway

[Flyway](https://flywaydb.org/) is a database migration tool that applies versioned SQL scripts in order. It tracks which migrations have already been applied in a `flyway_schema_history` table, so it only runs new ones.

### How it works

1. Flyway scans the `sql/` directory for migration files
2. Compares them against `flyway_schema_history` in the database
3. Applies any pending migrations in version order
4. Records each applied migration with a checksum

### File naming convention

```
V{version}__{description}.sql
```

- **V** — prefix indicating a versioned migration
- **{version}** — numeric version (e.g. `001`, `002`). Determines execution order
- **__** — double underscore separator (required)
- **{description}** — human-readable name using underscores (e.g. `create_raw_tables`)

Example: `V001__create_raw_tables.sql`

### Running migrations

Flyway runs automatically via Docker Compose when you `docker compose up`. It waits for PostgreSQL to be healthy, then applies pending migrations.

To run manually:

```bash
cd infra

# Apply pending migrations
docker compose run --rm flyway \
  -url=jdbc:postgresql://postgres:5432/vaultech \
  -user=vaultech -password=vaultech_dev migrate

# Check current status
docker compose run --rm flyway \
  -url=jdbc:postgresql://postgres:5432/vaultech \
  -user=vaultech -password=vaultech_dev info
```

### Adding new migrations

1. Create a new file in `migrations/sql/` following the naming convention (next version number)
2. Write plain SQL (DDL, DML, views, functions — anything PostgreSQL supports)
3. Run `docker compose up` or the manual `migrate` command above
4. Flyway applies only the new migration

### RDS compatibility

The same migration files work against AWS RDS — just change the JDBC URL, user, and password. Flyway can be run from any machine with network access to the RDS instance (no Docker required — Flyway is also available as a standalone CLI or Maven/Gradle plugin).

## Data Model

See [DATABASE.md](DATABASE.md) for the full schema documentation: tables, views, column descriptions, signal reference, data quality notes, and example queries.

## Seeding

Data is loaded via `scripts/seed.py` using PostgreSQL `COPY` (fast bulk insert). The script is idempotent — it skips tables that already contain data.

```bash
uv run python scripts/seed.py --env infra/.env
```
