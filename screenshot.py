from pyppeteer import launch
from ultralytics import YOLO
import asyncio

async def main(url):
    domain = url.replace('https://', '').replace('http://', '')
    name = f'./download/{domain}.png'
    browser = await launch()
    page = await browser.newPage()
    await page.goto(url)
    await page.setViewport({'width': 1200, 'height': 700})
    await page.screenshot({'path': name})
    await browser.close()
    return name



def get_label_name(id: int):
    dict_label = {0: 'logo'}
    return dict_label[id]

def get_logo(img_puch):
    model = YOLO('./best (1).pt')

    results = model.predict(img_puch)
    object_list = []
    object_label = []
    for result in results:
        # name = result.names
        boxes = result.boxes
        for box in boxes:
            cls = box.cls
            conf = box.conf
            object_list.append(str(conf.numpy()[0]))
            object_label.append(get_label_name(int(cls.numpy()[0])))
            # print(cls.numpy()[0])
    return object_label, object_list

img = asyncio.get_event_loop().run_until_complete(main('https://vk.com'))

#print(get_logo(img))
