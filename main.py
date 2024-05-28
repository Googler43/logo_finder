import ultralytics.data.loaders

image_path = "./logo/layer-19-48.png"
path = "./best.pt"
model_name = 'mega_category_test/1'

def get_label_name(id: int):
  dict_label = {0: 'mf_horizontal', 1: 'mf_logo', 2: 'mf_vertical'}
  return dict_label[id]

from ultralytics import YOLO
model = YOLO(path)
results = model([image_path], save=True)  # return a list of Results objects , save=True

object_list = []
object_label = []
for result in results:
    name = result.names
    boxes = result.boxes
    for box in boxes:
        cls = box.cls
        conf = box.conf
        object_list.append(str(conf.numpy()[0]))
        object_label.append(get_label_name(int(cls.numpy()[0])))
        #print(cls.numpy()[0])
print(object_list)
print(object_label)


