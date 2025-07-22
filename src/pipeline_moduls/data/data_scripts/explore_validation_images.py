import logging
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class ImageNetExplorer:
    """
    Diese Klasse dient der Exploration des ImageNet-Validierungsdatensatzes
    und zugehöriger Bounding-Box-Annotationen.
    """

    def __init__(self, image_tar_path: Path, bbox_tar_path: Path) -> None:
        """
        Initialisiert den Explorer mit den Pfaden zu den .tar-Dateien.

        Args:
            image_tar_path (Path): Pfad zur .tar-Datei mit den Validierungsbildern.
            bbox_tar_path (Path): Pfad zur .tgz-Datei mit den Bounding Box Annotations.
        """
        self.image_tar_path = image_tar_path
        self.bbox_tar_path = bbox_tar_path

    def explore_validation_images(self) -> int:
        """
        Analysiert die .tar-Datei mit den Validierungsbildern.

        Returns:
            int: Anzahl der Bilddateien im Archiv.
        """
        logger.info("Starte Analyse der Validierungsbilder")

        with tarfile.open(self.image_tar_path, "r") as tar:
            members = tar.getmembers()
            image_files = [m.name for m in members if m.isfile()]
            logger.info("Anzahl Bilddateien: %d", len(image_files))

            if image_files:
                logger.debug("Beispielfile: %s", image_files[0])
                logger.info("Namensschema erkannt: ILSVRC2012_val_XXXXXXXX.JPEG")

            return len(image_files)

    def explore_bounding_boxes(self) -> int:
        """
        Analysiert die .tgz-Datei mit den Bounding-Box-XML-Dateien.

        Returns:
            int: Anzahl der XML-Dateien im Archiv.
        """
        logger.info("Starte Analyse der Bounding Box Annotationen")

        with tarfile.open(self.bbox_tar_path, "r:gz") as tar:
            members = tar.getmembers()
            xml_files = [m for m in members if m.isfile() and m.name.endswith(".xml")]
            logger.info("Anzahl Bounding Box XML-Dateien: %d", len(xml_files))

            if xml_files:
                first_xml = xml_files[0]
                file = tar.extractfile(first_xml)
                if file is not None:
                    xml_content = file.read().decode("utf-8")
                    root = ET.fromstring(xml_content)
                    logger.debug("Dateiname: %s", first_xml.name)
                    logger.debug("XML Root Tag: %s", root.tag)

                    for child in root:
                        logger.debug("Tag: %s", child.tag)
                        if child.tag == "object":
                            for obj_child in child:
                                logger.debug("  %s: %s", obj_child.tag, obj_child.text)
                                if obj_child.tag == "bndbox":
                                    for coord in obj_child:
                                        logger.debug(
                                            "    %s: %s", coord.tag, coord.text
                                        )
                            break

            return len(xml_files)


def main() -> None:
    """
    Hauptfunktion zur Ausführung der Datensatz-Exploration.
    Führt Pfadprüfung und Analyse durch.
    """
    project_root = Path(__file__).resolve().parents[2]
    img_tar = project_root / "data" / "raw" / "ILSVRC2012_img_val.tar"
    bbox_tar = project_root / "data" / "raw" / "ILSVRC2012_bbox_val_v3.tgz"

    logger.info("Starte ImageNet Validierungsdatensatz-Analyse")

    if not img_tar.exists():
        logger.error("❌ Images TAR nicht gefunden: %s", img_tar)
        return

    if not bbox_tar.exists():
        logger.error("❌ Bounding Box TAR nicht gefunden: %s", bbox_tar)
        return

    logger.info("✅ Images TAR gefunden: %s", img_tar)
    logger.info("✅ Bounding Box TAR gefunden: %s", bbox_tar)

    explorer = ImageNetExplorer(img_tar, bbox_tar)

    num_images = explorer.explore_validation_images()
    num_bboxes = explorer.explore_bounding_boxes()

    logger.info("=== ZUSAMMENFASSUNG ===")
    logger.info("Anzahl Validierungsbilder: %d", num_images)
    logger.info("Anzahl Bounding Box Dateien: %d", num_bboxes)

    logger.info("=== NÄCHSTE SCHRITTE ===")
    logger.info("1. Extrahiere TAR-Dateien")
    logger.info("2. Erstelle Mapping Bild ↔ Bounding Box")
    logger.info("3. Baue Dataset Interface für Pipeline")


if __name__ == "__main__":
    main()
