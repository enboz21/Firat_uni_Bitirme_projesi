import os, json, boto3, requests
from datetime import datetime
from botocore.config import Config # Tek eklenen kütüphane bu
from pathlib import Path

# --- AYARLAR ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = PROJECT_ROOT / "veriCekme+maskeleme"
GEOJSON_DIR, HAVUZ_DIR = PIPELINE_DIR / "geojsons", PIPELINE_DIR / "outputs"
# Bu ayar donmayı engeller: 120 saniye yanıt gelmezse bağlantıyı keser.
S3_CONF = Config(connect_timeout=60, read_timeout=120, retries={'max_attempts': 3})
S3_CRED = {
    'aws_access_key_id': os.getenv('CDSE_S3_ACCESS_KEY'),
    'aws_secret_access_key': os.getenv('CDSE_S3_SECRET_KEY'),
    'endpoint_url': 'https://eodata.dataspace.copernicus.eu',
}

def get_config():
    conf = {}
    config_path = PIPELINE_DIR / "ayarlar.txt"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            for l in f:
                if "=" in l: k, v = l.strip().split("=", 1); conf[k.strip()] = v.strip()
    return conf

# --- ANA İŞLEM ---
os.makedirs(HAVUZ_DIR, exist_ok=True)
cfg = get_config()
if not S3_CRED['aws_access_key_id'] or not S3_CRED['aws_secret_access_key']:
    raise RuntimeError(
        'CDSE_S3_ACCESS_KEY ve CDSE_S3_SECRET_KEY ortam değişkenlerini ayarlayın. '
        'Örnek kullanım README.md dosyasında yer alır.'
    )
s3 = boto3.client('s3', config=S3_CONF, **S3_CRED)
filtreler = [f.strip() for f in cfg['FILTRELER'].split(",")]

for geo_file in os.listdir(GEOJSON_DIR):
    if not geo_file.endswith(".geojson"): continue
    
    print(f"> {geo_file} taranıyor...")
    with open(os.path.join(GEOJSON_DIR, geo_file), 'r', encoding='utf-8') as f: d = json.load(f)
    c = d['features'][0]['geometry']['coordinates'][0] if d['type'] == 'FeatureCollection' else d['coordinates'][0]
    wkt = "POLYGON ((" + ", ".join([f"{p[0]} {p[1]}" for p in c]) + "))"

    # 1. API Sorgusu
    q = (f"$filter=Collection/Name eq 'SENTINEL-2' and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') "
         f"and contains(Name,'MSIL2A') and ContentDate/Start gt {datetime.strptime(cfg['BASLANGIC_TARIHI'], '%d.%m.%Y').isoformat()}Z "
         f"and ContentDate/Start lt {datetime.strptime(cfg['BITIS_TARIHI'], '%d.%m.%Y').isoformat()}Z "
         f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value le {cfg['BULUT_ORANI']})")
    
    res = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?{q}&$top=1000&$expand=Attributes").json().get('value', [])
    if not res: continue

    # 2. En İyi Günü Seç (Mantık: En çok benzersiz kare + En az bulut)
    gunler = {}
    for item in res:
        tarih, grid = item['ContentDate']['Start'].split('T')[0], item['Name'].split('_')[5]
        bulut = next((a['Value'] for a in item['Attributes'] if a['Name'] == 'cloudCover'), 100)
        if tarih not in gunler: gunler[tarih] = {}
        if grid not in gunler[tarih] or bulut < gunler[tarih][grid]['cloud']:
            gunler[tarih][grid] = {'data': item, 'cloud': bulut}

    best_date = max(gunler, key=lambda d: (len(gunler[d]), -sum(x['cloud'] for x in gunler[d].values())))
    secilenler = [x['data'] for x in gunler[best_date].values()]
    print(f"  Seçilen: {best_date} ({len(secilenler)} dosya)")

    # 3. İndir
    for urun in secilenler:
        s3_prefix = urun['S3Path'].replace("/eodata/", "")
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket='eodata', Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                if any(f in obj['Key'] for f in filtreler):
                    yerel = os.path.join(HAVUZ_DIR, obj['Key'].split("/")[-1])
                    if os.path.exists(yerel) and os.path.getsize(yerel) == obj['Size']:
                        print(f"  [VAR] {os.path.basename(yerel)}")
                    else:
                        print(f"  [İNDİRİLİYOR] {os.path.basename(yerel)}")
                        try:
                            s3.download_file('eodata', obj['Key'], yerel)
                        except Exception as e:
                            print(f"  [HATA] İndirilemedi: {e}")
                            if os.path.exists(yerel): os.remove(yerel)

print("Bitti.")
