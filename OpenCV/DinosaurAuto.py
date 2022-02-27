import base64
from collections import deque
from io import BytesIO



import cv2
import numpy as np
from PIL import ImageGrab
from PIL import Image
import pyautogui as auto
from skimage.metrics import structural_similarity
from skimage.morphology import reconstruction
import time
import csv
import gym
import os
from gym import error, spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import selenium
import imageio
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
import stable_baselines3
from stable_baselines3 import PPO
def csvPlot(data):
    file = open("Data.csv", 'w')
    typeWriter = csv.writer(file)
    typeWriter.writerow(data)
def RestartGame(gameImage, gameXPos, gameYPos):
    restartTemplate = cv2.imread("Template/RestartButton.png", 0)
    restartHeight, restartWidth = restartTemplate.shape
    gameHeight, gameWidth = gameImage.shape
    detectedImage = cv2.matchTemplate(gameImage, restartTemplate, cv2.TM_CCOEFF_NORMED)
    # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(detectedImage)
    # topLeft = maxLoc
    # bottomRight = (topLeft[0] + clockWidth, topLeft[0] + clockHeight)
    # cv2.rectangle(roomImage, topLeft, bottomRight, 255, 2)
    threshold = 0.9
    # cv2.imshow("RoomGray", roomImageGray)
    # cv2.imshow("RoomColor", roomImage)
    # cv2.imshow("ClockGray", clockImage)
    # cv2.waitKey(0)
    location = np.where(detectedImage >= threshold)
    for point in zip(*location[::-1]):
        auto.click(gameXPos + point[0] + restartWidth / 2, gameYPos + point[1] + restartHeight / 2)
        auto.keyUp("down")
        auto.keyDown("up")
        time.sleep(0.5)
        auto.keyUp("up")
        return True
    return False
def MethodOne():
    fileData = []
    gameSpeed = 0 #pixels per frame
    lastFrameEnd = 0
    lastFrameTime = 0
    deltaTime = 1
    endOffset = 3
    jumpDistance = 170
    timeOffset = 1.7
    minSize = 5
    minGap = 20
    dimension = (1132, 379, 1732, 505)
    i = 0
    while True:
        #jumpDistance
        i += 1
        currentGap = 0
        cactusEnd = 0
        currentSize = 0
        cactusIsSelected = False
        screen = ImageGrab.grab(bbox = dimension)
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2GRAY)
        height, width = screenNP.shape
        #edges = cv2.Canny(blur, 25, 100)
        #gameSpeed is always increasing
        r, screenNP = cv2.threshold(screenNP, 125, 255, cv2.THRESH_BINARY_INV)
        for pixel in range(0, width - 75, 1):
            if (cactusIsSelected):
                if (currentGap >= minGap):
                    cactusEnd = pixel + 75 - currentGap + endOffset
                    deltaTime = time.time() - lastFrameTime
                    break
                elif (screenNP[int(height / 1.1)][pixel + 75] == 255):
                    currentGap = 0
                    cactusIsSelected = False
                else:
                    currentGap += 1
            elif (screenNP[int(height / 1.1)][pixel + 75] == 255):
                currentSize += 1
            elif (currentSize >= minSize):
                cactusIsSelected = True
            screenNP[int(height / 1.1)][pixel + 75] = 125
        #if (screenNP[int(height / 1.5)][45] < 100 or screenNP[int(height / 1.5)][115] < 100):
            #auto.hotkey("space")
        if (cactusIsSelected and (cactusEnd - jumpDistance <= 0)):
            auto.press("space")
        else:
            if (lastFrameEnd >= cactusEnd):
                gameSpeed = (lastFrameEnd - cactusEnd) / deltaTime
                print(gameSpeed)
                fileData.append(str(gameSpeed))
            lastFrameEnd = cactusEnd
            lastFrameTime = time.time()
        cv2.imshow("Haise Sasaki", screenNP)
        #cv2.imwrite("Original/" + str(i) + ".png", screenNP)
        if cv2.waitKey(1) & 0xff == ord("q"):
            csvPlot(fileData)
            break
def MethodTwo():
    fileData = []
    gameSpeed = 0  # pixels per frame
    lastFrameEnd = 0
    lastFrameTime = 0
    deltaTime = 1
    endOffset = 3
    jumpDistance = 170
    timeOffset = 1.7
    minSize = 5
    minGap = 20
    dimension = (1132, 379, 1732, 505)
    i = 0
    while True:
        # jumpDistance
        i += 1
        currentGap = 0
        cactusEnd = 0
        currentSize = 0
        cactusIsSelected = False
        screen = ImageGrab.grab(bbox=dimension)
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2GRAY)
        height, width = screenNP.shape
        # edges = cv2.Canny(blur, 25, 100)
        # gameSpeed is always increasing
        r, screenNP = cv2.threshold(screenNP, 125, 255, cv2.THRESH_BINARY_INV)
        for pixel in range(0, width - 75, 1):
            if (cactusIsSelected):
                if (currentGap >= minGap):
                    cactusEnd = pixel + 75 - currentGap + endOffset
                    deltaTime = time.time() - lastFrameTime
                    lastFrameTime = time.time()
                    break
                elif (screenNP[int(height / 1.1)][pixel + 75] == 255):
                    currentGap = 0
                    cactusIsSelected = False
                else:
                    currentGap += 1
            elif (screenNP[int(height / 1.1)][pixel + 75] == 255):
                currentSize += 1
            elif (currentSize >= minSize):
                cactusIsSelected = True
            screenNP[int(height / 1.1)][pixel + 75] = 125
        # if (screenNP[int(height / 1.5)][45] < 100 or screenNP[int(height / 1.5)][115] < 100):
        # auto.hotkey("space")
        if (lastFrameEnd >= cactusEnd and cactusIsSelected):
            if (gameSpeed < (lastFrameEnd - cactusEnd) / deltaTime):
                gameSpeed = (lastFrameEnd - cactusEnd) / deltaTime
                print(gameSpeed)
                fileData.append(str(gameSpeed))
        lastFrameEnd = cactusEnd
        if (cactusIsSelected and (cactusEnd - gameSpeed / 2 <= 0)):
            auto.press("space")

        cv2.imshow("Haise Sasaki", screenNP)
        # cv2.imwrite("Original/" + str(i) + ".png", screenNP)
        if cv2.waitKey(1) & 0xff == ord("q"):
            csvPlot(fileData)
            break
def MethodThree():
    fileData = []
    gameSpeed = 0  # pixels per frame
    lastFrameEnd = 0
    lastFrameTime = 0
    deltaTime = 1
    endOffset = 3
    jumpDistance = 175
    startDistance = jumpDistance
    timeOffset = 1.75
    minSize = 5
    minGap = 20
    dimension = (1132, 379, 1732, 505)
    i = 0
    second = time.time()
    startTime = time.time()
    MAX_SPEED_TIME = -1
    while True:
        if (time.time() - second >= 1):
            jumpDistance += timeOffset
            second = time.time()
        # jumpDistance
        i += 1
        currentGap = 0
        cactusEnd = 0
        currentSize = 0
        cactusIsSelected = False
        screen = ImageGrab.grab(bbox=dimension)
        screenNP = np.array(screen)
        screenNP = cv2.cvtColor(screenNP, cv2.COLOR_BGR2GRAY)
        height, width = screenNP.shape
        # edges = cv2.Canny(blur, 25, 100)
        # gameSpeed is always increasing
        r, screenNP = cv2.threshold(screenNP, 125, 255, cv2.THRESH_BINARY_INV)
        for pixel in range(0, width - 100, 1):
            if (cactusIsSelected):
                if (currentGap >= minGap):
                    cactusEnd = pixel + 100 - currentGap + endOffset
                    if (cactusEnd < width):
                        cv2.line(screenNP, (cactusEnd, 0), (cactusEnd, height), (125, 125, 125), thickness = 1)
                    break
                elif (screenNP[int(height / 1.15)][pixel + 100] == 255):
                    currentGap = 0
                    cactusIsSelected = False
                else:
                    currentGap += 1
            elif (screenNP[int(height / 1.15)][pixel + 100] == 255):
                currentSize += 1
            elif (currentSize >= minSize):
                cactusIsSelected = True
            #screenNP[int(height / 1.15)][pixel + 100] = 125
        # if (screenNP[int(height / 1.5)][45] < 100 or screenNP[int(height / 1.5)][115] < 100):
        # auto.hotkey("space")
        if (cactusIsSelected and (cactusEnd - jumpDistance <= 0)):
            auto.keyUp("down")
            auto.press("space")
            time.sleep(0.2)#* startDistance / jumpDistance
            auto.keyDown("down")
        cv2.imshow("Haise Sasaki", screenNP)
        if (RestartGame(screenNP, dimension[0], dimension[1])):
            jumpDistance = startDistance
            second = time.time()
        # cv2.imwrite("Original/" + str(i) + ".png", screenNP)
        if cv2.waitKey(1) & 0xff == ord("q"):
            csvPlot(fileData)
            break
class ChromeTRex(gym.Env):
    def __init__(self, screenWidth, screenHeight, chromeDriverPath):
        self.screen_width = screenWidth
        self.screen_height = screenHeight
        self.chromedriver_path = chromeDriverPath
        self.num_observations = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (self.screen_width, self.screen_height, 4),
            dtype = np.uint8
        )
        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("--mute-audio")
        _chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(
            executable_path = self.chromedriver_path,
            options = _chrome_options
        )
        self.current_key = None
        self.state_queue = deque(maxlen = 4)
        self.actions_map = [
            Keys.ARROW_RIGHT,
            Keys.ARROW_UP,
            Keys.ARROW_DOWN
        ]
        actionChains = ActionChains(self._driver)
        self.keydown_actions = [actionChains.key_down(i) for i in self.actions_map]
        self.keyup_actions = [actionChains.key_up(i) for i in self.actions_map]
    def _get_image(self):
        pretext = "data:image/png;base64,"
        image = self._driver.execute_script("return document.querySelector('canvas.runner-canvas').toDataURL()")
        image = image[len(pretext):]
        return np.array(Image.open(BytesIO(base64.b64decode(image))))
    def _next_observation(self):
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2GRAY)
        image = image[:500, :500]
        image = cv2.resize(image, (self.screen_width, self.screen_height))
        self.num_observations += 1
        self.state_queue.append(image)
        if (len(self.state_queue) < 4):
            return np.stack([image] * 4, axis = -1)
        else:
            return np.stack(self.state_queue, axis = -1)
        return image
    def _get_done(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    def _get_score(self):
        try:
            score = int(''.join(self._driver.execute_script("return Runner.instance_.distanceMeter.digits")))
        except:
            score = 0
        return score
    def step(self, action:int):
        self._driver.find_element_by_tag_name("body").send_keys(self.actions_map[action])
        observation = self._next_observation()
        done = self._get_done()
        if (done):
            reward = -1
        else:
            reward = 1
        time.sleep(0.015)
        return observation, reward, done, {"score":self._get_score()}
    def reset(self):
        try:
            self._driver.get("chrome://dino")
        except WebDriverException as e:
            print(e)
        WebDriverWait(self._driver, 10).until(
            EC.presence_of_element_located((
                By.CLASS_NAME,
                "runner-canvas"
            ))
        )
        self._driver.find_element_by_tag_name("body").send_keys(Keys.SPACE)
        return self._next_observation()
    def render(self, mode:str):
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2GRAY)
        return image
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    #def ReinforcementLearning():

    #ReinforcementLearning()
def Main():
    environmentLambda = lambda: ChromeTRex(
        100,
        100,
        webdriver.Chrome(ChromeDriverManager().install())
    )
    env = ChromeTRex(
        100,
        100,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver")
    )
    train = True
    saveDirectory = "WilliamDinoModel.zip"
    #environment = SubprocVecEnv([environmentLambda for i in range(1)])
    if train:
        model = PPO('MlpPolicy', env, verbose = 1)
        model.learn(total_timesteps = 1e6)
        '''checkpointCallback = CheckpointCallback(
            save_freq = 1000,
            save_path = "Checkpoints/",
            name_prefix = saveDirectory,
        )
        model = PPO(
            CnnPolicy,
            environment,
            verbose = 1,
            tensorboard_log = "Logs/"
        )
        model.learn(
            total_timesteps = 5000000,
            callback = [checkpointCallback]
        )'''
        model.save(saveDirectory)
    model = PPO.load(saveDirectory, env = env)
    images = []
    ovs = env.reset()
    image = model.env.render("a")
    for i in tqdm(range(500)):
        images.append(image)
        action, states = model.predict(ovs, deterministic = True)
        ovs, rewards, dones, info = env.step(action)
        image = model.env.render()
    imageio.mimsave('William.gif', [np.array(image) for i, image in enumerate(images)], fps = 10)
Main()
#MethodThree()

