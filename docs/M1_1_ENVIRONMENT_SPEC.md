# M1.1 Environment Spec

## Data Entities

### suppliers (6 records)
- supplier_id, name, rating (0-5), region, certified (0/1), delivery_reliability (0-1), description

### products (24 records)
- product_id, supplier_id (FK), name, category, price (>0), memory_gb, delivery_days, stock, warranty_months, model_number, description

### episodes
- episode_id, task_id, status (active/submitted/verified/failed), created_at, completed_at

### procurement_submissions
- submission_id, episode_id (FK), task_id, decision_type (select_product/no_solution), product_id (FK, nullable), quantity, justification, submitted_at

## Database Schema
- SQLite with foreign keys enforced (PRAGMA foreign_keys = ON)
- CHECK constraints on price, stock, rating, certified, decision_type

## Page Routes
| Route | Method | Purpose |
|---|---|---|
| /health | GET | Health check with DB stats |
| /smoke | GET | M1.0 smoke test page |
| / | GET | Home — task list |
| /tasks/{id} | GET | Task detail |
| /tasks/{id}/start | POST | Create episode |
| /products | GET | Product listing with filters |
| /products/{id} | GET | Product detail |
| /suppliers/{id} | GET | Supplier detail |
| /procurement/new | GET | Procurement form |
| /procurement/submit | POST | Submit procurement |
| /procurement/result/{id} | GET | Submission result |

## Episode Lifecycle
1. POST /tasks/{id}/start → creates active episode
2. Browse products, select or declare no_solution
3. POST /procurement/submit → episode becomes submitted
4. Verifier checks → verified or failed

## Environment Reset
- `reset-db`: drops all tables, recreates schema, re-seeds from JSON
- Idempotent: same seed data produces identical DB state
- Only allows resetting project database paths

## DOM Stability Rules
- All interactive elements have stable data-testid attributes
- No random DOM IDs
- Form submission uses standard GET/POST
- No JavaScript async race conditions
- No infinite scroll or lazy loading
