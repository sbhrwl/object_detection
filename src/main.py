from read_image import load_image
from get_model_labels_and_output_layers import get_model_labels_and_output_layers
from object_detection import detect_persons

if __name__ == '__main__':
    frame = load_image()
    model, LABELS, ln = get_model_labels_and_output_layers()
    print(model)
    print(LABELS)
    print(ln)
    detect_persons(frame,
                   model,
                   ln,
                   person_idx=LABELS.index("person"))
