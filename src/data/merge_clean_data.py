import os
import shutil

def copy_images(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for file in os.listdir(src_dir):
        if file.endswith('.tif') or file.endswith('.png') or file.endswith('.jpg'):
            src_file = os.path.join(src_dir, file)
            # To avoid name collisions, we can prefix the filename if needed
            # But the original files probably have unique names. Let's just prefix with the source folder name if we want, or just check collision
            base, ext = os.path.splitext(file)
            dest_file = os.path.join(dest_dir, file)
            # Simple collision handling
            idx = 1
            while os.path.exists(dest_file):
                dest_file = os.path.join(dest_dir, f"{base}_{idx}{ext}")
                idx += 1
                
            shutil.copy2(src_file, dest_file)
            count += 1
    return count

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    processed_dir = os.path.join(base_dir, "data", "processed")
    pseudo_dir = os.path.join(base_dir, "data", "pseudo_label")
    output_dir = os.path.join(base_dir, "data", "processed_v4")

    print("Copying Clean Train Data...")
    y_tr1 = copy_images(os.path.join(processed_dir, "train", "young"), os.path.join(output_dir, "train", "young"))
    s_tr1 = copy_images(os.path.join(processed_dir, "train", "senescent"), os.path.join(output_dir, "train", "senescent"))
    y_tr2 = copy_images(os.path.join(pseudo_dir, "young"), os.path.join(output_dir, "train", "young"))
    s_tr2 = copy_images(os.path.join(pseudo_dir, "senescent"), os.path.join(output_dir, "train", "senescent"))
    
    print("Copying Clean Test Data...")
    y_te1 = copy_images(os.path.join(processed_dir, "test", "young"), os.path.join(output_dir, "test", "young"))
    s_te1 = copy_images(os.path.join(processed_dir, "test", "senescent"), os.path.join(output_dir, "test", "senescent"))

    print("\n--- SUMMARY ---")
    print(f"TRAIN - Young: {y_tr1} (processed) + {y_tr2} (pseudo) = {y_tr1+y_tr2}")
    print(f"TRAIN - Senescent: {s_tr1} (processed) + {s_tr2} (pseudo) = {s_tr1+s_tr2}")
    print(f"TEST - Young: {y_te1} (processed)")
    print(f"TEST - Senescent: {s_te1} (processed)")
    print("All quarantine (qaran) data was ignored.")
