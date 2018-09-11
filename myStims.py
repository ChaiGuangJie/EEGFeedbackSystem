from psychopy import visual, event, core
from psychopy.visual import shape, circle
import math
import random

__all__ = ['Fixation','RightArrow','LeftArrow','Xaxis','Yaxis',
           'CountDown','featureStim','DrawTextStim','WaitOneKeyPress','TargetWindow','Bullet']

class ClueStim():
    def __init__(self, win, vertices, lineWidth=1.5):
        self.win = win
        self.clue = shape.ShapeStim(win,
                                    units='pix',
                                    vertices=vertices,
                                    lineWidth=lineWidth,
                                    closeShape=False,
                                    )

    def draw(self, duration):
        self.clue.draw()
        self.win.flip()
        core.wait(duration)
        self.win.flip()

    def startDraw(self, flip=True):
        self.clue.autoDraw = True
        if flip:
            self.win.flip()

    def endDraw(self, flip=True):
        self.clue.autoDraw = False
        if flip:
            self.win.flip()


def Fixation(win, radius,x=0,y=0):
    return ClueStim(win, vertices=((x, -radius+y), (x, radius+y),
                                   (x, y), (-radius+x, y), (radius+x, y)))


def RightArrow(win, radius):
    return ClueStim(win, vertices=((radius / 2, radius / 2),
                                   (radius, 0), (radius / 2, -radius / 2), (radius, 0), (0, 0)))


def LeftArrow(win, radius):
    return ClueStim(win, vertices=((-radius / 2, radius / 2),
                                   (-radius, 0), (-radius / 2, -radius / 2), (-radius, 0), (0, 0)))


def TargetWindow(win):
    half_x = win.size[0]/2
    half_y = win.size[1]/2
    centerX = 0
    centerY = 1.5 * half_y
    return ClueStim(win,vertices=((-half_x*0.70,half_y-1),(-0.65*half_x,0.2*-half_y),(0.65*half_x,0.2*-half_y),(half_x*0.7,half_y-1),(-half_x*0.7,half_y-1))),(centerX,centerY)

def Xaxis(win, radius=None,cutOut=0.9,y=0):
    if radius:
        halfAxis = radius
    else:
        halfAxis = cutOut * win.size[0] / 2.0
    return ClueStim(win, vertices=((-halfAxis, y), (halfAxis, y)), lineWidth=1)


def Yaxis(win, radius=None,cutOut=0.9,x=0):
    if radius:
        halfAxis = radius
    else:
        halfAxis = cutOut * win.size[1] / 2.0
    return ClueStim(win, vertices=((x, halfAxis), (x, -halfAxis)), lineWidth=1)


class CountDown():
    def __init__(self, win,duration=2):
        self.win = win
        self.degree = 2 * math.pi / 60 / duration
        self.radius = 20
        self.pointSegments = []

        for i in range(60*duration+1):
            x = self.radius * math.sin(i * self.degree)
            y = self.radius * math.cos(i * self.degree)
            self.pointSegments.append((x, y))

        self.pathSegments = []

        lastX = self.pointSegments[0][0]
        lastY = self.pointSegments[0][1]
        for (x, y) in self.pointSegments[1:]:
            path = shape.ShapeStim(win,
                                   units='pix',
                                   vertices=((lastX, lastY), (x, y)),
                                   lineWidth=2,
                                   closeShape=False,
                                   lineColor='white',
                                   autoDraw=False
                                   )
            lastX = x
            lastY = y
            self.pathSegments.append(path)

    def draw(self,slightDraw = False):  # todo 参数改为持续时间
        pathLen = len(self.pathSegments)
        if slightDraw:
            for s in self.pathSegments:
                s.draw()
                self.win.flip()
                # core.wait(refreshRate)  # todo 延时不准确，改为控制度数
        else:
            for ptr in range(pathLen):
                self._drawAllSegment(self.pathSegments[:ptr + 1])
                # core.wait(refreshRate)  # todo 延时不准确，改为控制度数

    def _drawAllSegment(self, allSegment):
        for s in allSegment:
            s.draw()
        self.win.flip()


class featureStim():
    def __init__(self, win, features=None, dotRaduis=5):
        self.win = win
        self.features = features
        self.dotRaduis = dotRaduis
        self.featureDots = []
        # self.positiveColor = 'red'
        # self.negativeColor = 'blue'
        self.colorDict = {
            -1: 'blue',
            1: 'red',
            2: 'green',
            3: 'yellow'
        }
        self.winScale = win.size.min() / 2.0
        if self.features:
            for (x, y, label) in features:
                dot = circle.Circle(
                    win,
                    radius=dotRaduis,
                    pos=(x * self.winScale, y * self.winScale),
                    lineWidth=0,
                    fillColor=self.colorDict[label],
                    opacity=0.1,
                    units='pix',
                )
                self.featureDots.append(dot)
#                    lineColor=self.colorDict[label],
    def startDrawAllFeatures(self, highlightLast=True, gradients = False):
        for p in self.featureDots[:-5]:
            p.autoDraw = True
            # p.draw()
        for p,o in zip(self.featureDots[-5:],[0.2,0.4,0.6,0.8,1.0]):
            p.autoDraw = True
            if gradients:
                p.opacity = o
        if highlightLast:
            highlightCircle = circle.Circle(self.win,
                                            radius=self.dotRaduis + 1,
                                            pos=self.featureDots[-1].pos,
                                            units='pix',
                                            lineWidth=2)
            highlightCircle.draw()
        self.win.flip()

    def endDrawAllFeatures(self):
        for p in self.featureDots:
            p.autoDraw = False
            p.opacity = 0.1
            # p.draw()
        self.win.flip()

    def drawNewFeature(self, feature):
        dot = circle.Circle(
            self.win,
            radius=self.dotRaduis,
            pos=(feature[0] * self.winScale, feature[1] * self.winScale),
            lineColor=self.colorDict[feature[-1]],
            fillColor=self.colorDict[feature[-1]],
            units='pix',
        )
        self._highlightDot(dot)
        self.win.flip()
        self.featureDots.append(dot)

    def _scaleDot(self, dot, scale, waitTime):
        dot.radius += scale
        dot.draw()
        self.win.flip()
        core.wait(waitTime)

    def _highlightDot(self, dot, repeatTimes=2, intervalTime=0.1):
        for times in range(repeatTimes):
            for _ in range(5):
                self._scaleDot(dot, 4, intervalTime)
            for _ in range(5):
                self._scaleDot(dot, -4, intervalTime)

    def removeLastFeature(self):
        lastFeature = self.featureDots.pop()
        lastFeature.autoDraw = False

def DrawTextStim(win,text):
    text = visual.TextStim(win,text=text)
    text.draw()
    win.flip()

def WaitOneKeyPress(win,key,textStim = None):
    if textStim is not None:
        text = visual.TextStim(win, text=textStim)
        text.draw()
        win.flip()
    allKeys = event.waitKeys(keyList=[key])
    for thisKey in allKeys:
        if thisKey == key:
            break

class Bullet():
    def __init__(self,win,targetCenter,dotRaduis =5, duration = 1):#targetCenter = (x,y)
        self.bulletList = []
        self.win = win
        # self.start_y = start_y
        self.targetCenter = targetCenter
        self.dotRaduis = dotRaduis
        self.duration = duration
        # self.velocity = initVelocity
        self.colorDict = {
            -1: 'blue',
            1: 'red',
            2: 'green',
            3: 'yellow'
        }
        self.startPoint = (0,-win.size[1]/2)
        # self.verticalDistance = math.sqrt(
        #     abs(targetCenter[0]-self.startPoint[0]) * abs(targetCenter[0]-self.startPoint[0])
        #     + abs(targetCenter[1] - self.startPoint[1]) * abs(targetCenter[1] - self.startPoint[1]))


    def add_new_bullet(self, x, y, label): #destination = (x,y)
        bullet = circle.Circle(
                    self.win,
                    radius=self.dotRaduis,
                    pos=self.startPoint,
                    lineWidth=0,
                    fillColor=self.colorDict[label],
                    units='pix',
                )
        # radiusDistance = math.sqrt(abs(x-self.targetCenter[0])*abs(x-self.targetCenter[0])+abs(y-self.targetCenter[1])*abs(y-self.targetCenter[1]))
        # realDistance = math.sqrt(
        #     abs(x-self.startPoint[0])*abs(x-self.startPoint[0])
        #     + abs(y-self.startPoint[1])*abs(y-self.startPoint[1]))
        # v_realDis = realDistance/self.duration
        # v_verticalDis = v_realDis * self.verticalDistance/realDistance
        # v_x = v_realDis * (x-self.startPoint[0])/realDistance
        # v_y = v_realDis * abs(y-self.targetCenter[1])/realDistance
        v_x = (x - self.startPoint[0])/self.duration #duration为小数的时候会严重影响计算落点的准确性！
        v_y = abs(y - self.startPoint[1])/self.duration
        self.bulletList.append({'bullet':bullet,
                                'x' : x,
                                'y' : y,
                                'v_x':v_x,
                                'v_y':v_y,
                                'arrived':False})

    def update_bullets(self,dt):
        # d = dt * self.velocity
        for b in self.bulletList:
            if not b['arrived']:
                dy = b['v_y'] * dt
                dx = b['v_x'] * dt
                b['bullet'].pos += (dx, dy)
                # if b['bullet'].pos[0] >= b['x']:
                #     b['bullet'].pos = (b['x'], b['bullet'].pos[1])
                # if b['bullet'].pos[1] >= b['y']:
                #     b['bullet'].pos = (b['bullet'].pos[0], b['y'])
                if  b['bullet'].pos[1] >= b['y']: #b['bullet'].pos[0] >= b['x'] and
                    print(b['bullet'].pos,(b['x'], b['y']))
                    b['bullet'].pos = (b['x'], b['y'])
                    b['arrived'] = True
                # todo 判断是否到达终点，如果到达，将pos改为终点值,并将arrived置为True
                # print(b['bullet'].pos[0])
            else:
                if b['bullet'].opacity > 0.2:
                    b['bullet'].opacity -= 0.01
            b['bullet'].draw()
        self.win.flip()




if __name__ == "__main__":
    win = visual.Window([1000, 800])
    event.globalKeys.add(key='escape', func=core.quit, name='esc')

    trial = 1

    def createFeatures():
        features = []
        for i in range(80):
            label = random.choice([-1, 1])
            x = random.gauss(label / 2.0, 0.2)
            # x = random.uniform(-1, 1)
            y = 0  # random.uniform(-1, 1)
            features.append((x, y, label))
        return features
    features = createFeatures()
    fs = featureStim(win, features=features, dotRaduis=10)

    while True:
        fixation = Fixation(win, 10)
        # fixation.startDraw()
        fixation.draw(2)

        def createRandomArrow():
            arrowDict = {
                1 : RightArrow(win,20),
                -1: LeftArrow(win,20)
            }
            arrow = random.choice([-1, 1])
            return arrowDict[arrow]

        # l = LeftArrow(win,20)
        # l.draw(3)
        # r = RightArrow(win,20)
        # r.draw(5)
        arrow  = createRandomArrow()
        arrow.draw(2)


        countDown = CountDown(win,duration=4)
        #transport.run()
        countDown.draw(slightDraw=False)
        #transport.pause()

        fixation.startDraw()
        x = Xaxis(win,radius=win.size[0]/2.0)
        y = Yaxis(win)
        x.startDraw()
        #y.startDraw()
        #y.draw(5)
        # x.endDraw()

        _label = random.choice([-1, 1])
        fs.drawNewFeature((random.gauss(_label/2.0,0.2), 0, _label)) #random.uniform(-1, 1)
        fs.startDrawAllFeatures(gradients=True)
        #core.wait(5)
        print('trial ',trial,' end')
        trial+=1

        allKeys = event.waitKeys(keyList=['left', 'right'])
        for thisKey in allKeys:
            if thisKey == 'left':
                fs.removeLastFeature()
            elif thisKey == 'right':
                break

        fs.endDrawAllFeatures()
        arrow.endDraw()
        x.endDraw()
        #y.endDraw()
        fixation.endDraw()

