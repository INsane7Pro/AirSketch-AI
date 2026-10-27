
import os
import urllib.request
import numpy as np
import cv2
import joblib
import time

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "quickdraw_data"
MODEL_DIR = "models"

# Choose 100 or 345.
NUMBER_OF_CLASSES = 100

# More samples improve training but take longer.
SAMPLES_PER_CLASS = 500

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Full official category list.
ALL_CLASSES = """
aircraft carrier,airplane,alarm clock,ambulance,angel
animal migration,ant,anvil,apple,arm
asparagus,axe,backpack,banana,bandage
barn,baseball bat,baseball,basket,basketball
bat,bathtub,beach,bear,beard
bed,bee,belt,bench,bicycle
binoculars,bird,birthday cake,blackberry,blueberry
book,boomerang,bottlecap,bowtie,bracelet
brain,bread,bridge,broccoli,broom
bucket,bulldozer,bus,bush,butterfly
cactus,cake,calculator,calendar,camel
camera,camouflage,campfire,candle,cannon
canoe,car,carrot,castle,cat
ceiling fan,cell phone,cello,chair,chandelier
church,circle,clarinet,clock,cloud
coffee cup,compass,computer,cookie,cooler
couch,cow,crab,crayon,crocodile
crown,cruise ship,cup,diamond,dishwasher
diving board,dog,dolphin,donut,door
dragon,dresser,drill,drums,duck
dumbbell,ear,elbow,elephant,envelope
eraser,eye,eyeglasses,face,fan
feather,fence,finger,fire hydrant,fireplace
firetruck,fish,flamingo,flashlight,flip flops
floor lamp,flower,flying saucer,foot,fork
frog,frying pan,garden hose,garden,giraffe
goatee,golf club,grapes,grass,guitar
hamburger,hammer,hand,harp,hat
headphones,hedgehog,helicopter,helmet,hexagon
hockey puck,hockey stick,horse,hospital,hot air balloon
hot dog,hot tub,hourglass,house plant,house
hurricane,ice cream,jacket,jail,kangaroo
key,keyboard,knee,knife,ladder
lantern,laptop,leaf,leg,light bulb
lighter,lighthouse,lightning,line,lion
lipstick,lobster,lollipop,mailbox,map
marker,matches,megaphone,mermaid,microphone
microwave,monkey,moon,mosquito,motorbike
mountain,mouse,moustache,mouth,mug
mushroom,nail,necklace,nose,octagon
onion,oven,owl,paintbrush,paint can
palm tree,panda,pants,paper clip,parachute
parrot,passport,peanut,penguin,pencil
penguin,piano,pickup truck,pig,pillow
pineapple,pizza,pliers,police car,pond
pool,popsicle,postcard,potato,power outlet
purse,rabbit,raccoon,radio,rain
rainbow, rake,remote control,rhinoceros,rifle
river,roller coaster,rollerskates, sailboat,sandwich
saw,saxophone,school bus,scissors,scorpion
screwdriver,sea turtle,see saw,shark,sheep
shoe,shorts,shovel,sink,skateboard
skull, skyscraper,sleeping bag,smiley face,smoke
snail,snake,snorkel,snowflake,snowman
soccer ball,sock,socks,solar system,spoon
spreadsheet,square,squiggle,squirrel, stairs
star,steak,stereo,stethoscope,stitches
stop sign,stove,strawberry,streetlight,string bean
submarine,suitcase,sun,swan,sweater
swing set,sword,syringe,t-shirt,table
teapot,teddy-bear,telephone,television,tennis racquet
tent,the Eiffel Tower,the Great Wall of China,the Mona Lisa,tiger
toaster,toe,toilet,tooth, toothbrush
toothpaste,tornado,tractor,traffic light,train
tree,triangle,trombone,truck,trumpet
umbrella,underwear,van,vase,violin
washing machine,watermelon,waterslide,whale,wheel
windmill,wine bottle,wine glass, wristwatch,yoga
zebra,zigzag
"""

CLASSES = [
    item.strip().replace(" ", "_")
    for item in ALL_CLASSES.replace("\n", ",").split(",")
    if item.strip()
]

# Keep category names consistent with QuickDraw filenames.
# Use only the first requested number of classes.
CLASSES = CLASSES[:NUMBER_OF_CLASSES]

X = []
y = []


def extract_features(image):
    image = cv2.resize(image, (64, 64))

    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


for index, label in enumerate(CLASSES, start=1):
    # The official filenames use spaces, not underscores.
    actual_name = label.replace("_", " ")
    filename = os.path.join(DATA_DIR, label + ".npy")

    url_name = urllib.parse.quote(actual_name, safe="")
    url = (
        "https://storage.googleapis.com/quickdraw_dataset/"
        "full/numpy_bitmap/" + url_name + ".npy"
    )

    if not os.path.exists(filename):
        print(f"[{index}/{len(CLASSES)}] Downloading {actual_name}...")
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as error:
            print("Download failed:", actual_name, error)
            continue

    try:
        drawings = np.load(filename, mmap_mode="r")
        count = min(SAMPLES_PER_CLASS, len(drawings))

        rng = np.random.default_rng(42)
        indices = rng.choice(len(drawings), count, replace=False)

        for sample_index in indices:
            image = drawings[sample_index].reshape(28, 28).astype(np.uint8)

            # Convert to white strokes on black background.
            image = cv2.threshold(
                image, 127, 255, cv2.THRESH_BINARY
            )[1]

            X.append(extract_features(image))
            y.append(actual_name)

        print(f"Loaded {count} drawings: {actual_name}")

    except Exception as error:
        print("Could not process:", actual_name, error)


if len(X) == 0:
    raise RuntimeError(
        "No training samples loaded. Check your internet connection "
        "and category download URLs."
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print("\nTraining model...")
start = time.time()

model = LinearSVC(
    C=1.0,
    max_iter=5000,
    dual="auto",
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

joblib.dump(
    {
        "model": model,
        "encoder": encoder,
        "classes": list(encoder.classes_),
    },
    os.path.join(MODEL_DIR, "drawing_model.joblib"),
)

print("\nTraining completed.")
print(f"Actual categories loaded: {len(encoder.classes_)}")
print(f"Validation accuracy: {accuracy:.2%}")
print(f"Training time: {time.time() - start:.1f} seconds")
print("Model saved to models/drawing_model.joblib")
