# Optional: Bildnamen extrahieren, falls verfügbar
#                 image_names = None
#                 if hasattr(dataloader.dataset, 'get_sample_info'):
#                     # Versuche, Bildnamen für Logging zu holen
#                     indices = range(batch_idx * dataloader.batch_size, (batch_idx + 1) * dataloader.batch_size)
#                     image_names = [
#                         dataloader.dataset.get_sample_info(i).get('image_path', f'image_{i}')
#                         for i in indices
#                     ]
