import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Lista de URLs de índices
URLS = [
    "http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html",
    "http://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html",
    "http://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html",
    "http://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html",
    "http://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html",
    "http://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/index.html",
]

OUTPUT_DIR = "mass_roads_downloads"


def download_image(img_url, save_dir, session: requests.Session):
    os.makedirs(save_dir, exist_ok=True)
    filename = img_url.split("/")[-1]
    path = os.path.join(save_dir, filename)

    if os.path.exists(path):
        print(f"Ya existe: {filename}")
        return

    print(f"Descargando {filename} ...")
    try:
        response = session.get(img_url, stream=True, timeout=20)
    except requests.RequestException as e:
        print(f"Error de red al descargar {img_url}: {e}")
        return

    if response.status_code == 200:
        with open(path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Error al descargar {img_url}: {response.status_code}")


def process_index(url, session: requests.Session):
    print(f"\n📥 Procesando índice: {url}")

    # Carpeta local de destino (train/map, test/sat, etc.)
    path_parts = url.split("mass_roads/")[1].split("/index.html")[0]
    save_dir = os.path.join(OUTPUT_DIR, path_parts)

    # Descargar index.html
    try:
        html_resp = session.get(url, timeout=20)
        html_resp.raise_for_status()
        html = html_resp.text
    except requests.RequestException as e:
        print(f"No se pudo obtener el índice {url}: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")

    # Extensiones válidas
    valid_ext = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    links = soup.find_all("a")

    total = 0
    skipped = 0
    for link in links:
        href = link.get("href")
        if href and href.lower().endswith(valid_ext):
            img_url = urljoin(url, href)
            total += 1
            filename = img_url.split("/")[-1]
            path = os.path.join(save_dir, filename)
            if os.path.exists(path):
                skipped += 1
                print(f"Ya existía y se saltó: {filename}")
                continue
            download_image(img_url, save_dir, session)

    if total:
        print(f"Resumen índice: {total} archivos, {skipped} ya existentes, {total - skipped} nuevos.")


def main():
    # Reutilizar sesión con cabecera para evitar bloqueos del servidor
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })

    for url in URLS:
        process_index(url, session)

    print("\n✔ Descarga completada.")


if __name__ == "__main__":
    main()