from tkinter import *
import random, copy
import numpy as np

def bound(n, small, big):
    return max(min (n, big), small)

def sigmoid(x):
    return 1/(1 + np.e ** (-x)) 

def almostEqual(a, b, epsilon = 10**(-3)):
    return abs(a-b) < epsilon

def distance(a, b):
    return (a**2 + b**2)**0.5

def distanceFromLine(x1, y1, x2, y2, x0, y0):
    return abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2 * y1 - y2 * x1)/distance(x2-x1, y2-y1)

class Circle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def draw(self, canvas, xStart):
        canvas.create_oval(self.x - self.r - xStart, self.y - self.r, self.x + self.r - xStart, self.y + self.r, fill = self.color)

    def collision(self, other, height, viewX):
        return not 0 < self.y < height or self.x < viewX or (distance((self.x - other.x), (self.y - other.y)) <= self.r + other.r)


class Sensor(object):
    def __init__(self, relativeTheta, sensorLength):
        self.relativeTheta = relativeTheta
        self.sensorLength = sensorLength
        self.tripped = False

    def updateAll(self, cX, cY, carTheta):
        self.updateBegPos(cX, cY)
        self.updateTheta(carTheta)
        self.updateEndPos()

    def updateBegPos(self, cX, cY): #cX, cY is x y of car
        self.x1, self.y1 = cX, cY

    def updateTheta(self, carTheta):
        self.theta = carTheta + self.relativeTheta

    def updateEndPos(self):
        self.x2, self.y2 = self.sensorLength * np.array([np.cos(self.theta), np.sin(self.theta)]) + np.array([self.x1, self.y1])

    def detects(self, obstacle):
        onLine = distanceFromLine(self.x1, self.y1, self.x2, self.y2, obstacle.x, obstacle.y) <= obstacle.r
        xMin = min(self.x1, self.x2) - obstacle.r
        xMax = max(self.x1, self.x2) + obstacle.r
        yMin = min(self.y1, self.y2) - obstacle.r
        yMax = max(self.y1, self.y2) + obstacle.r
        inXRange = xMin < obstacle.x < xMax
        inYRange = yMin < obstacle.y < yMax 
        return onLine and inXRange and inYRange

    def detectsOutOfBounds(self, height):
        return not 0 < self.y2 < height

    def draw(self, canvas, xStart):
        color = "green" if self.tripped else "black"
        canvas.create_line(self.x1 - xStart, self.y1, self.x2 - xStart, self.y2, fill = color)

class Car(Circle): #car radius = 10
    def __init__(self, x, y, v, sensorSpread = np.pi/3, numSensors = 6, sensorLength = 65, syn0 = None, syn1 = None, deltaTheta = None): 
        super().__init__(x, y, 10) #radius hardcoded to 10
        self.color = "red"
        self.vMag = v
        self.vTheta = random.random() * np.pi - np.pi/2
        self.deltaTheta = 0.3 if deltaTheta == None else deltaTheta

        
        #Sensor Init Math
        self.sensorLength = sensorLength
        self.sensorSpread = sensorSpread
        self.numSensors = numSensors
        self.lap = 0
        self.maxX = self.x

        sensorSep = self.sensorSpread/(self.numSensors - 1)
        sensorStart = -self.sensorSpread/2
        self.sensors = []
        for i in range(numSensors):
            relativeTheta = sensorStart + i * sensorSep
            newSensor = Sensor(relativeTheta, self.sensorLength)
            newSensor.updateAll(self.x, self.y, self.vTheta)
            self.sensors.append(newSensor)

        self.weightFile = "weights/" + (self.numSensors) + "SensorWeights.npz"
        if syn0 == syn1 == None:
            try:
                self.trainFromFile()
                print ("Loaded " + self.weightFile)
            except:
                print ("Something went wrong with npz file, using random weighting instead")
                self.randomSyn()
        else:
            self.theseSyn(syn0, syn1)

        
        self.initialValues = (x, y, v)

    def clickedOn(self, event, viewXStart):
        x, y = event.x + viewXStart, event.y
        if distance((self.x - x), (self.y - y)) <= self.r:
            sensors = np.array([self.sensorSpread, self.sensorLength])
            np.savez(self.weightFile, self.syn0, self.syn1, sensors)
            print ("Written to " + self.weightFile)
        

    def trainFromFile(self):
        weights = np.load(self.weightFile)
        self.syn0 = weights["arr_0"]
        self.syn1 = weights["arr_1"]
        self.others = weights["arr_2"]

    def randomSyn(self):
        self.syn0 = 2*np.random.random((self.numSensors,5)) - 1
        self.syn1 = 2*np.random.random((5,3)) - 1

    def theseSyn(self, syn0, syn1):
        self.syn0 = syn0
        self.syn1 = syn1

    def getExact(self):
        return Car(*self.initialValues, self.sensorSpread, self.numSensors, self.sensorLength, self.syn0, self.syn1, self.deltaTheta)

    def getMutations(self, num):
        newCars = [self.getExact()]
        while len(newCars) < num:
            newCars.append(self.mutate())
        return newCars

    def mutate(self):
        mutationSize = 0.7
        minDeltaTheta, maxDeltaTheta = 0.2, 0.7
        sensorSpreadMutationChoices = [0]*3 + [random.random() - 0.5]
        sensorSpreadRange = (0.2, 2 * np.pi)
        sensorLengthMutationChoices = [0]*10 + list(range(-10, 11))
        sensorLengthRange = (40, 120)
        while (True):
            newSyn0 = self.syn0 + mutationSize * (np.random.rand(*self.syn0.shape) - 0.5)
            newSyn1 = self.syn1 + mutationSize * (np.random.rand(*self.syn1.shape) - 0.5)
            newDeltaTheta = bound(self.deltaTheta + (random.random() - 0.5) * 0.33, minDeltaTheta, maxDeltaTheta)
            newSensorSpread = bound(self.sensorSpread + random.choice(sensorSpreadMutationChoices), *sensorSpreadRange)
            newSensorLength = bound(self.sensorLength + random.choice(sensorLengthMutationChoices), *sensorLengthRange)
            car = Car(*self.initialValues, newSensorSpread, self.numSensors, newSensorLength, newSyn0, newSyn1, newDeltaTheta)
            if car.verify():
                return car
        
    def update(self, obstacles, height): #returns true if car needs destroyed
        self.move()
        self.getInputFromSensors(obstacles, height)
        self.convertSensorInputToSteering()
    
    def getTrippedSensors(self):
        sensorsTripped = []
        for i in range(len(self.sensors)):
            sensorsTripped.append(1 if self.sensors[i].tripped else 0)
        return sensorsTripped

    def convertSensorInputToSteering(self):
        l0 = np.array([self.getTrippedSensors()])
        l1 = sigmoid(np.dot(l0, self.syn0))
        l2 = sigmoid(np.dot(l1, self.syn1))[0]
        output = list(l2).index(max(l2))
        self.accelerate(output)

    def verify(self):
        l0 = np.array([0] * self.numSensors)
        l1 = sigmoid(np.dot(l0, self.syn0))
        l2 = sigmoid(np.dot(l1, self.syn1))
        output = list(l2).index(max(l2))
        return output == 1

    def getVx(self):
        return self.vMag * np.cos(self.vTheta)

    def getVy(self):
        return self.vMag * np.sin(self.vTheta)

    def seekEnd(self): #proportional controller to set theta to 0
        target = 0
        proportionalConstant = 10
        self.vTheta += (target - self.vTheta) / proportionalConstant

    def accelerate(self, d):
        if d == 0: #left
            self.vTheta -= self.deltaTheta

        elif d == 1: #seek end
            self.seekEnd()

        elif d == 2: #right
            self.vTheta += self.deltaTheta


    def draw(self, canvas, xStart):
        super().draw(canvas, xStart)
        for sensor in self.sensors:
            sensor.draw(canvas, xStart)

    def move(self):
        self.y += self.getVy()
        self.x += self.getVx()
        for sensor in self.sensors:
            sensor.updateAll(self.x, self.y, self.vTheta)
        self.maxX = max(self.maxX, self.x)

    def getPossibleObstacles(self, obstacles):
        possibleObstacles = []
        for obstacle in obstacles:
            if (distance(self.x - obstacle.x, self.y - obstacle.y) <= self.r + obstacle.r + self.sensorLength + 3): # 3 is fudge factor
                possibleObstacles.append(obstacle)
        return possibleObstacles

    def getInputFromSensors(self, obstacles, height):
        possibleObstacles = self.getPossibleObstacles(obstacles)
        sensorsTripped = [0] * self.numSensors
        for sensor in self.sensors:    
            sensor.tripped = False
            if sensor.detectsOutOfBounds(height):
                self.vTheta *= -1
                break
            else:
                for obstacle in possibleObstacles:
                    if sensor.detects(obstacle):
                        sensor.tripped = True
                        break

class Obstacle(Circle):
    def __init__(self, x, y, r):
        super().__init__(x, y, r)
        self.color = "blue"

class Animation(object):
    def __init__(self, screenWidth, screenHeight):
        self.paused = False
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        self.carsToMutate = 3
        self.obstacles = []
        self.width = 5000
        self.height = self.screenHeight
        self.numObstacles = 250
        self.numCars = 30
        self.generation = 0
        """
        count = True

        for i in range(100, self.width, self.width//30):
            if count:
                self.obstacles.append(Obstacle(i, 3*self.height/5, 75))
                self.obstacles.append(Obstacle(i, self.height/5, 75))
                self.obstacles.append(Obstacle(i, self.height, 75))
            else:
                self.obstacles.append(Obstacle(i, 0, 75))
                self.obstacles.append(Obstacle(i, 2*self.height/5, 75))
                self.obstacles.append(Obstacle(i, 4*self.height/5, 75))
            count = not count



        """
        #self.obstacles.append(Obstacle(225, self.height/2, 50))
        while len(self.obstacles) < self.numObstacles:
            obstacleR = random.randint(10, 20)
            obstacleY = random.randint(obstacleR, self.height - obstacleR)
            obstacleX = random.randint(obstacleR + 50, self.width - obstacleR)
            self.obstacles.append(Obstacle(obstacleX, obstacleY, obstacleR))
        self.reinit()

    def reinit(self, cars = None):
        self.generation += 1
        self.timerCount = 0
        if cars == None:
            self.cars = []
            while len(self.cars) < self.numCars:
                car = Car(0, random.randint(self.screenHeight/3, 2*self.screenHeight/3), 4)
                if car.verify():
                    self.cars.append(car)
        else:
            self.cars = copy.deepcopy(cars)
        self.allCars = self.cars
        self.updateView()

    def mutateBestCars(self):
        originalCars = 0
        newCars = []
        for car in self.cars[:self.carsToMutate]:
            newCars.extend(car.getMutations((self.numCars-originalCars)//self.carsToMutate))
        while (len(newCars) < self.numCars):
            car = Car(0, self.height/2, 4)
            if car.verify():
                newCars.append(car)
        self.reinit(newCars)

    def updateView(self):
        if (len(self.cars) != 0):
            carX = max(car.x for car in self.cars)
            self.viewXStart = carX - self.screenWidth/2

    def mousePressed(self, event):
        for car in self.cars:
            car.clickedOn(event, self.viewXStart)

    def keyPressed(self, event):
        if event.keysym == "Left":
            self.viewXStart -= 10
        elif event.keysym == "Right":
            self.viewXStart += 10
        elif event.char == "p":
            self.paused = not self.paused

    def timerFired(self):
        if not self.paused:
            deadCars = set()
            for i in range(len(self.cars)):
                car = self.cars[i]
                if car.update(self.obstacles, self.height):
                    deadCars.add(i)

            for i in range(len(self.cars)):
                car = self.cars[i]
                for obstacle in self.obstacles:
                    if car.collision(obstacle, self.height, self.viewXStart):
                        deadCars.add(i)

            if len(self.cars) + len(deadCars) <= self.carsToMutate: #reinitialize
                self.mutateBestCars()

            deadCars = sorted(list(deadCars))
            for deadCar in deadCars[::-1]:
                self.cars.pop(deadCar)
                
            self.updateView()

    def redrawAll(self, canvas):
        for obstacle in self.obstacles:
            obstacle.draw(canvas, self.viewXStart)

        for car in self.cars:
            car.draw(canvas, self.viewXStart)

        canvas.create_text(10, 10, text = self.viewXStart + self.screenWidth/2, anchor = W)
        canvas.create_text(self.screenWidth, 10, text = "Cars Remaining: " + str(len(self.cars)), anchor = E)
        canvas.create_text(self.screenWidth/2, 10, text = "Generation " + str(self.generation))

    ####################################
    # use the run function as-is
    ####################################
    def redrawAllWrapper(self, canvas):
            canvas.delete(ALL)
            self.redrawAll(canvas)
            canvas.update()    

    def mousePressedWrapper(self, event, canvas):
        self.mousePressed(event)
        self.redrawAllWrapper(canvas)

    def keyPressedWrapper(self, event, canvas):
        self.keyPressed(event)
        self.redrawAllWrapper(canvas)

    def timerFiredWrapper(self, canvas):
        self.timerFired()
        self.redrawAllWrapper(canvas)
        # pause, then call timerFired again
        canvas.after(self.timerDelay, self.timerFiredWrapper, canvas)

    def run(self):
        self.timerDelay = 30 # milliseconds

        # create the root and the canvas
        root = Tk()
        canvas = Canvas(root, width=self.screenWidth, height=self.screenHeight)
        canvas.pack()
        # set up events
        root.bind("<Button-1>", lambda event:
                                self.mousePressedWrapper(event, canvas))
        root.bind("<Key>", lambda event:
                                self.keyPressedWrapper(event, canvas))
        self.timerFiredWrapper(canvas)
        # and launch the app
        root.mainloop()  # blocks until window is closed
        print("bye!")

Animation(1200, 600).run()
