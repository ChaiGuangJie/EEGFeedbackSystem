from psychopy import visual, event, core
from psychopy.visual import shape, circle
import math
import random


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


def Fixation(win, radius):
    return ClueStim(win, vertices=((0, -radius), (0, radius),
                                   (0, 0), (-radius, 0), (radius, 0)))


def RightArrow(win, radius):
    return ClueStim(win, vertices=((radius / 2, radius / 2),
                                   (radius, 0), (radius / 2, -radius / 2), (radius, 0), (0, 0)))


def LeftArrow(win, radius):
    return ClueStim(win, vertices=((-radius / 2, radius / 2),
                                   (-radius, 0), (-radius / 2, -radius / 2), (-radius, 0), (0, 0)))


def Xaxis(win, radius=None):
    if radius:
        halfAxis = radius
    else:
        halfAxis = win.size.min() * 0.9 / 2.0
    return ClueStim(win, vertices=((-halfAxis, 0), (halfAxis, 0)), lineWidth=1)


def Yaxis(win, radius=None):
    if radius:
        halfAxis = radius
    else:
        halfAxis = win.size.min() * 0.9 / 2.0
    return ClueStim(win, vertices=((0, halfAxis), (0, -halfAxis)), lineWidth=1)


class CountDown():
    def __init__(self, win):
        self.win = win
        self.degree = 2 * math.pi / 60
        self.radius = 20
        self.pointSegments = []

        for i in range(61):
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

    def draw(self, duration=0.06,slightDraw = False):  # todo 参数改为持续时间
        if slightDraw:
            for s in self.pathSegments:
                s.draw()
                self.win.flip()
                core.wait(duration)  # todo 延时不准确，改为控制度数

        else:
            pathLen = len(self.pathSegments)
            for ptr in range(pathLen):
                self._drawAllSegment(self.pathSegments[:ptr + 1])
                core.wait(duration)  # todo 延时不准确，改为控制度数

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
        self.positiveColor = 'red'
        self.negativeColor = 'blue'
        self.colorDict = {
            1: self.positiveColor,
            -1: self.negativeColor
        }
        self.winScale = win.size.min() / 2.0
        if self.features:
            for (x, y, label) in features:
                dot = circle.Circle(
                    win,
                    radius=dotRaduis,
                    pos=(x * self.winScale, y * self.winScale),
                    lineColor=self.colorDict[label],
                    fillColor=self.colorDict[label],
                    units='pix',
                )
                self.featureDots.append(dot)

    def startDrawAllFeatures(self, highlightLast=True):
        for p in self.featureDots:
            p.autoDraw = True
            # p.draw()
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


if __name__ == "__main__":
    win = visual.Window([1920, 1080])
    event.globalKeys.add(key='escape', func=core.quit, name='esc')

    trial = 1

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


        countDown = CountDown(win)
        countDown.draw(slightDraw=False)

        fixation.startDraw()
        x = Xaxis(win)
        y = Yaxis(win)
        x.startDraw()
        y.startDraw()
        #y.draw(5)
        # x.endDraw()


        def createFeatures():
            features = []
            for i in range(80):
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                label = random.choice([-1, 1])
                features.append((x, y, label))
            return features

        fs = featureStim(win, features=createFeatures())
        fs.drawNewFeature((random.uniform(-1, 1), random.uniform(-1, 1), random.choice([-1, 1])))
        fs.startDrawAllFeatures()
        #core.wait(5)
        print('trial ',trial,' end')
        trial+=1

        event.waitKeys(keyList=['space'])

        fs.endDrawAllFeatures()
        arrow.endDraw()
        x.endDraw()
        y.endDraw()
        fixation.endDraw()

