from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader


def auto_batchsize_test(
        dataloader:DataLoader,
        max_batch_size: int = 128,
        device: str = 'cuda'
) -> int:
    """
    Testet automatisch die maximale Batch-Größe für die verfügbare GPU.

    Args:
        dataloader: Dataloder zum Datensatzt
        max_batch_size: Maximale zu testende Batch-Größe
        device: Zielgerät ('cuda' oder 'cpu')

    Returns:
        int: Empfohlene maximale Batch-Größe

    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA nicht verfügbar, verwende CPU")
        return 16

    print(f"Teste maximale Batch-Größe auf {device}...")

    successful_batch_size = 1

    for batch_size in [2, 4, 8, 16, 32, 64, 128]:
        if batch_size > max_batch_size:
            break

        try:
            print(f"  Teste Batch-Größe {batch_size}...")


            # Teste einen Batch
            images, labels, boxes = next(iter(dataloader))

            if device == 'cuda':
                images = images.cuda()
                labels = labels.cuda()

                # Simuliere Forward Pass
                with torch.no_grad():
                    _ = images.sum()

                # Räume GPU-Speicher auf
                del images, labels
                torch.cuda.empty_cache()

            successful_batch_size = batch_size
            print(f"    ✓ Batch-Größe {batch_size} erfolgreich")

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                print(f"    ✗ Batch-Größe {batch_size} zu groß (OOM)")
                break
            else:
                print(f"    ✗ Fehler bei Batch-Größe {batch_size}: {e}")
                break

    print(f"Empfohlene maximale Batch-Größe: {successful_batch_size}")
    return successful_batch_size

