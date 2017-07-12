from tkinter import *
import random
import numpy as np

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

    def draw(self, canvas):
        canvas.create_oval(self.x - self.r, self.y - self.r, self.x + self.r, self.y + self.r, fill = self.color)

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

    def draw(self, canvas):
        color = "green" if self.tripped else "black"
        canvas.create_line(self.x1, self.y1, self.x2, self.y2, fill = color)

class Car(Circle): #car radius = 10
    def __init__(self, x, y, v, sensorSpread = np.pi/2, numSensors = 7, sensorLength = 80): 
        super().__init__(x, y, 10) #radius hardcoded to 10
        self.color = "red"
        self.vMag = v
        self.vTheta = 0
        
        #Sensor Init Math
        self.sensorLength = sensorLength
        self.sensorSpread = sensorSpread
        self.numSensors = numSensors

        sensorSep = self.sensorSpread/(self.numSensors - 1)
        sensorStart = -self.sensorSpread/2
        self.sensors = []
        for i in range(numSensors):
            relativeTheta = sensorStart + i * sensorSep
            newSensor = Sensor(relativeTheta, self.sensorLength)
            newSensor.updateAll(self.x, self.y, self.vTheta)
            self.sensors.append(newSensor)

        self.syn0 = 2*np.random.random((7,4)) - 1 #sensors to hidden layer
        self.syn1 = 2*np.random.random((4,2)) - 1 #hidden layer to steering

    
    def getTrippedSensors(self):
        sensorsTripped = []
        for i in range(len(self.sensors)):
            sensorsTripped.append(1 if self.sensors[i].tripped else 0)
        return sensorsTripped


    def convertSensorInputToSteering(self):
        l0 = np.array([self.getTrippedSensors()])
        l1 = sigmoid(np.dot(l0, self.syn0))
        l2 = sigmoid(np.dot(l1, self.syn1))[0]

        if l2[0] < 0.1:
            self.accelerate(0)
        if l2[1] < 0.1:
            self.accelerate(1)
        

    def getVx(self):
        return self.vMag * np.cos(self.vTheta)

    def getVy(self):
        return self.vMag * np.sin(self.vTheta)

    def collision(self, other):
        return (distance((self.x - other.x), (self.y - other.y)) <= self.r + other.r)

    def accelerate(self, d):
        deltaTheta = 0.2

        if d == 0: #left
            self.vTheta -= deltaTheta

        elif d == 1: #right
            self.vTheta += deltaTheta

    def draw(self, canvas):
        super().draw(canvas)
        for sensor in self.sensors:
            sensor.draw(canvas)

    def move(self):
        self.y += self.getVy()
        for sensor in self.sensors:
            sensor.updateAll(self.x, self.y, self.vTheta)

    def getPossibleObstacles(self, obstacles):
        possibleObstacles = []
        for obstacle in obstacles:
            if (distance(self.x - obstacle.x, self.y - obstacle.y) <= self.r + obstacle.r + self.sensorLength + 3): # 3 is fudge factor
                possibleObstacles.append(obstacle)
        return possibleObstacles

    def getInputFromSensors(self, obstacles):
        possibleObstacles = self.getPossibleObstacles(obstacles)
        sensorsTripped = [0] * self.numSensors
        for sensor in self.sensors:
            sensor.tripped = False
            for obstacle in possibleObstacles:
                if sensor.detects(obstacle):
                    sensor.tripped = True
                    break


class Obstacle(Circle):
    def __init__(self, x, y, r):
        super().__init__(x, y, r)
        self.color = "blue"

    def move(self, carVx):
        self.x -= carVx

class Animation(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []
        self.obstacleFreqControl = 8
        self.timerCount = 0
        self.car = Car(self.width/4, self.height/2, 3)
        self.cars = [self.car]
        self.gameOver = False
        self.score = 0

    def mousePressed(self, event):
        pass

    def keyPressed(self, event):
        if event.keysym == "Left":
            self.car.accelerate(0)
        elif event.keysym == "Right":
            self.car.accelerate(1)

    def timerFired(self):
        carVx = max(car.getVx() for car in self.cars)
        print (carVx)
        if not self.gameOver:
            self.score += carVx
            self.timerCount += 1
            if self.timerCount == self.obstacleFreqControl:
                self.timerCount = 0
                self.obstacles.append(Obstacle(self.width, random.randint(0, self.height), random.randint(5, 10)))
                
            for obstacle in self.obstacles:
                obstacle.move(carVx)
                if obstacle.x <= 0:
                    self.obstacles.remove(obstacle)
                if self.car.collision(obstacle):
                    self.gameOver = True
            self.car.move()
            self.car.getInputFromSensors(self.obstacles)
            self.car.convertSensorInputToSteering()


    def redrawAll(self, canvas):
        for obstacle in self.obstacles:
            obstacle.draw(canvas)
        self.car.draw(canvas)
        canvas.create_text(10, 10, text = "Fitness: " + str(self.score), anchor = W)

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
        self.timerDelay = 25 # milliseconds

        # create the root and the canvas
        root = Tk()
        canvas = Canvas(root, width=self.width, height=self.height)
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

Animation(800, 600).run()