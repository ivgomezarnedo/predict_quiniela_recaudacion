import json
from datetime import datetime
from scrapping import matches_scrapping, quiniela_resultados
from utils import db_management, s3_management, telegram_methods
import calculations
import ssl
import time

#TODO: Calcular probabilidad final para cada una de las quinielas con partidos en la BD (para poder compararlas).

def lambda_handler(event, context):
    current_timestamp = str(datetime.now())[0:19]
    telegram_methods.send_to_telegram("New lambda execution at " + current_timestamp)
    start = time.perf_counter()
    ssl._create_default_https_context = ssl._create_unverified_context
    print("Downloading DB from S3...")
    s3_management.get_db_from_s3()  # Download SQLite DB from S3
    quiniela_resultados.main()
    matches_scrapping.main()
    print("Uploading DB to S3...")
    db_management.close_sqlite_connection()
    s3_management.upload_db_to_s3()  # Upload SQLite DB to S3 with all changes persisted
    print("Calculating best Quiniela and sending it to Telegram...")
    print(f'Total execution duration: {time.perf_counter() - start}')
    calculations.main()
    return {
        'statusCode': 200,
        'body': json.dumps('Scrapping successfully finished!')
    }


#lambda_handler(None, None)
