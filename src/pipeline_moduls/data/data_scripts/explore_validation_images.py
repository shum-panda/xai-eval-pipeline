import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def explore_validation_images(tar_path):
    """Schaue was in der Validation TAR ist"""
    print("=== VALIDATION IMAGES ===")
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()

        print(f"Anzahl Dateien: {len(members)}")

        # Erste paar Dateinamen
        image_files = [m.name for m in members if m.isfile()][:10]
        print("Erste 10 Dateinamen:")
        for img in image_files:
            print(f"  {img}")

        # Naming pattern prüfen
        if image_files:
            print("\nNaming Pattern erkannt:")
            print("  Format: ILSVRC2012_val_XXXXXXXX.JPEG")

    return len([m for m in members if m.isfile()])


def explore_bounding_boxes(bbox_tar_path):
    """Schaue was in der Bounding Box TAR ist"""
    print("\n=== BOUNDING BOX ANNOTATIONS ===")

    with tarfile.open(bbox_tar_path, "r:gz") as tar:
        members = tar.getmembers()

        print(f"Anzahl Dateien: {len(members)}")

        # Erste paar Dateinamen
        bbox_files = [m.name for m in members if m.isfile()][:10]
        print("Erste 10 Dateinamen:")
        for bbox in bbox_files:
            print(f"  {bbox}")

        # Schaue dir eine beispiel XML an
        xml_files = [m for m in members if m.name.endswith(".xml")]
        if xml_files:
            print("\nBeispiel XML Struktur:")

            # Extrahiere erste XML
            first_xml = xml_files[0]
            xml_content = tar.extractfile(first_xml).read().decode("utf-8")

            # Parse XML
            root = ET.fromstring(xml_content)
            print(f"  Dateiname in XML: {first_xml.name}")
            print(f"  XML Root: {root.tag}")

            # Schaue dir die Struktur an
            for child in root:
                print(f"    {child.tag}: {child.text if child.text else 'hat Kinder'}")
                if child.tag == "object":
                    for obj_child in child:
                        print(f"      {obj_child.tag}: {obj_child.text}")
                        if obj_child.tag == "bndbox":
                            for coord in obj_child:
                                print(f"        {coord.tag}: {coord.text}")
                    break

    return len([m for m in members if m.isfile()])


def main():
    # Pfade zu deinen Dateien - angepasst für deine Struktur
    # Script liegt in xai-pipeline/src/data/, Daten in xai-pipeline/data/
    project_root = (
        Path(__file__).resolve().parents[2]
    )  # Gehe 2 Ebenen hoch zu xai-pipeline/
    img_tar = project_root / "data" / "raw" / "ILSVRC2012_img_val.tar"
    bbox_tar = project_root / "data" / "raw" / "ILSVRC2012_bbox_val_v3.tgz"

    print("ImageNet Validation Dataset Exploration")
    print("=" * 50)

    # Prüfe ob Dateien existieren
    if not img_tar.exists():
        print(f"❌ Images TAR nicht gefunden: {img_tar}")
        return

    if not bbox_tar.exists():
        print(f"❌ Bounding Box TAR nicht gefunden: {bbox_tar}")
        return

    print(f"✅ Images TAR: {img_tar}")
    print(f"✅ Bounding Box TAR: {bbox_tar}")

    # Exploration
    num_images = explore_validation_images(img_tar)
    num_bboxes = explore_bounding_boxes(bbox_tar)

    print("\n=== SUMMARY ===")
    print(f"Validation Images: {num_images}")
    print(f"Bounding Box Files: {num_bboxes}")

    # Nächste Schritte
    print("\n=== NÄCHSTE SCHRITTE ===")
    print("1. Extrahiere Validation Images")
    print("2. Extrahiere Bounding Box XMLs")
    print("3. Erstelle Mapping zwischen Bildern und Bounding Boxes")
    print("4. Baue Dataset Interface für deine Pipeline")


if __name__ == "__main__":
    main()
