$env:DATABASE_URL = 'postgresql://neondb_owner:npg_SzFm50IMjLkc@ep-twilight-cell-ak7nytds-pooler.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
cd C:\Users\bojun\Desktop\test\biotech-terminal
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
