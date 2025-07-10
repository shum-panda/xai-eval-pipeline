import logging
import tarfile
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def extract_validation_images(tar_path: Path, output_dir: Path, logger):
    """Extrahiere Validation Images aus TAR"""
    logger.info("Extrahiere Validation Images...")
    logger.info(f"Von: {tar_path}")
    logger.info(f"Nach: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with tarfile.open(tar_path, "r") as tar:
        members = tar.getmembers()

        for member in members:
            if member.isfile() and member.name.endswith(".JPEG"):
                # Extrahiere direkt in output_dir (ohne Unterordner)
                member.name = Path(member.name).name  # Nur Dateiname, kein Pfad
                tar.extract(member, path=output_dir)
                extracted_count += 1

                if extracted_count % 5000 == 0:
                    logger.info(f"  Extrahiert: {extracted_count} Bilder...")

    logger.info(f"✅ {extracted_count} Validation Images extrahiert")
    return extracted_count


def extract_bounding_boxes(bbox_tar_path: Path, output_dir: Path, logger):
    """Extrahiere Bounding Box XMLs aus TAR.GZ"""
    logger.info("Extrahiere Bounding Box Annotations...")
    logger.info(f"Von: {bbox_tar_path}")
    logger.info(f"Nach: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with tarfile.open(bbox_tar_path, "r:gz") as tar:
        members = tar.getmembers()

        for member in members:
            if member.isfile() and member.name.endswith(".xml"):
                # Extrahiere direkt in output_dir (ohne Unterordner)
                member.name = Path(member.name).name  # Nur Dateiname, kein Pfad
                tar.extract(member, path=output_dir)
                extracted_count += 1

                if extracted_count % 1000 == 0:
                    logger.info(f"  Extrahiert: {extracted_count} XMLs...")

    logger.info(f"✅ {extracted_count} Bounding Box XMLs extrahiert")
    return extracted_count


def main():
    logger = setup_logging()

    # Pfade für deine Projektstruktur
    project_root = Path(__file__).resolve().parents[2]  # xai-pipeline/

    # Input files
    img_tar = project_root / "data" / "raw" / "ILSVRC2012_img_val.tar"
    bbox_tar = project_root / "data" / "raw" / "ILSVRC2012_bbox_val_v3.tgz"

    # Output directories
    images_output = project_root / "data" / "extracted" / "validation_images"
    bbox_output = project_root / "data" / "extracted" / "bounding_boxes"

    logger.info("ImageNet Data Extraction")
    logger.info("=" * 50)
    logger.info(f"Projekt Root: {project_root}")

    # Validiere Input files
    if not img_tar.exists():
        logger.error(f"❌ Images TAR nicht gefunden: {img_tar}")
        return 1

    if not bbox_tar.exists():
        logger.error(f"❌ Bounding Box TAR nicht gefunden: {bbox_tar}")
        return 1

    logger.info(f"✅ Images TAR: {img_tar}")
    logger.info(f"✅ Bounding Box TAR: {bbox_tar}")

    try:
        # Extrahiere Images
        num_images = extract_validation_images(img_tar, images_output, logger)

        # Extrahiere Bounding Boxes
        num_bboxes = extract_bounding_boxes(bbox_tar, bbox_output, logger)

        # Summary
        logger.info("=" * 50)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"📁 Validation Images: {num_images}")
        logger.info(f"   Location: {images_output}")
        logger.info(f"📦 Bounding Boxes: {num_bboxes}")
        logger.info(f"   Location: {bbox_output}")
        return 0

    except Exception as e:
        logger.error(f"❌ Fehler bei Extraktion: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
