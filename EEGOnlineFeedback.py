from psychopy import visual, event, core
from psychopy.visual import circle, shape, line
import random,math
win = visual.Window([900, 900], monitor="testMonitor", units="cm")
#dot = visual.DotStim(win,dotSize=4.0)
fixation = shape.ShapeStim(win,
                           vertices=((0, -0.2), (0, 0.2), (0, 0), (-0.2, 0), (0.2, 0)),
                           lineWidth=1,
                           closeShape=False,
                           lineColor='white',
                           autoDraw=False
                           )
verLine = line.Line(win, (0, 0.8), (0, -0.8), units='norm',
                    lineColor='white', lineWidth=0.002, autoDraw=False)
horiLine = line.Line(win, (-0.8, 0), (0.8, 0), units='norm',
                     lineColor='white', lineWidth=0.002, autoDraw=False)

circle1 = circle.Circle(
    win,
    radius=0.01,
    pos=(
        0.5,
        0.3),
    lineColor='red',
    fillColor='red',
    units='norm')

circle2 = circle.Circle(win, radius=0.01, pos=(-0.6, -0.4),
                        lineColor='blue', fillColor='blue', units='norm')

event.globalKeys.add(
    key='q',
    modifiers=['ctrl'],
    func=core.quit,
    name='shutdown')
event.globalKeys.add(key='escape', func=core.quit, name='esc')

allPoint = []


def _scalePoint(p, scale, waitTime):
    p.radius += scale
    p.draw()
    win.flip()
    core.wait(waitTime)


def highlightPoint(p, repeatTimes=2, intervalTime=0.1):
    radius = p.radius.copy()
    #print('radius', radius)
    p.draw()
    win.flip()
    core.wait(intervalTime)

    for times in range(repeatTimes):
        for _ in range(5):
            _scalePoint(p, 0.006, intervalTime)
        for _ in range(5):
            _scalePoint(p, -0.006, intervalTime)
        # for scale in [0.01,0.012,0.014,0.016,0.018]:
        #     _scalePoint(p,scale,intervalTime)
        #p.radius = radius
        #print('inhighlight', p.radius)
    #p.radius += 0.005
    whiteCircle = circle.Circle(win,radius=0.014,pos=p.pos,units='norm')
    whiteCircle.draw()
    #win.flip()


def printAllPoint(allPoint):
    for i in allPoint:
        print(i.radius)
    print(len(allPoint))

def drawAllPoint(allPoint,autoDraw):
    for p in allPoint:
        p.autoDraw = autoDraw
        p.draw()
    win.flip()

def drawNewPoint(newDot):
    global allPoint
    if newDot.label == 1:
        color = 'red'
    else:
        color = 'blue'
    newCircle = circle.Circle(
        win,
        radius=0.01,
        pos=newDot.pos,
        lineColor=color,
        fillColor=color,
        units='norm')
    allPoint.append(newCircle)
    #print('inDraw', allPoint[-1].radius)
    # printAllPoint(allPoint)
    drawAllPoint(allPoint,autoDraw = True)

    highlightPoint(newCircle)
    #print('afterDraw', allPoint[-1].radius)
    drawAllPoint(allPoint,autoDraw = False)



class featureDot:
    def __init__(self, pos, label):
        self.pos = pos,
        self.label = label


for i in range(80):
    newDot = featureDot(pos=(random.uniform(-1, 1),
                             random.uniform(-1, 1)), label=random.randint(-1, 1))
    if newDot.label == 1:
        color = 'red'
    else:
        color = 'blue'
    newCircle = circle.Circle(
        win,
        radius=0.01,
        pos=newDot.pos,
        lineColor=color,
        fillColor=color,
        units='norm',
        interpolate=True
    )
    allPoint.append(newCircle)


degree = 2 * math.pi/60
radius = 20
segments = []
for i in range(61):
    x = radius * math.sin(i * degree)
    y = radius * math.cos(i * degree)
    segments.append((x, y))
lastX = segments[0][0]
lastY = segments[0][1]

allSegment = []
def drawAllSegment(allSegment):
    for s in allSegment:
        s.draw()
    win.flip()

rightArrow = shape.ShapeStim(win,
                           units='pix',
                            vertices=((radius/2, radius/2), (radius, 0), (radius/2, -radius/2), (radius, 0), (0, 0)),
                            lineWidth=1,
                            closeShape=False,
                            lineColor='white',
                            )
leftArrow = shape.ShapeStim(win,
                           units='pix',
                            vertices=((-radius/2, radius/2), (-radius, 0), (-radius/2, -radius/2), (-radius, 0), (0, 0)),
                            lineWidth=1,
                            closeShape=False,
                            lineColor='white',
                            )

def drawArrow():
    arrow = random.randint(-1, 1)
    if arrow == 1:
        rightArrow.draw()
    else:
        leftArrow.draw()
    win.flip()
    core.wait(2)



while True:
    allSegment = []
    fixation.autoDraw = False
    verLine.autoDraw = False
    horiLine.autoDraw = False

    fixation.draw()
    win.flip()
    core.wait(1)

    drawArrow()

    for (x, y) in segments[1:]:
        s = shape.ShapeStim(win,
                            units='pix',
                            vertices=((lastX, lastY), (x, y)),
                            lineWidth=2,
                            closeShape=False,
                            lineColor='white',
                            autoDraw=False
                            )
        allSegment.append(s)
        # s.draw()
        # win.flip()
        #########################
        # s.draw()
        # win.flip()
        #########################
        drawAllSegment(allSegment)
        lastX = x
        lastY = y
        core.wait(0.06)  # 根据单试次的长度调整

    fixation.autoDraw = True
    fixation.draw()

    verLine.autoDraw = True
    verLine.draw()
    horiLine.autoDraw = True
    horiLine.draw()

    newDot = featureDot(pos=(random.uniform(-1, 1),
                             random.uniform(-1, 1)), label=random.randint(-1, 1))
    drawNewPoint(newDot)
    #core.wait(0.3)

    #drawAllPoint(allPoint)

    core.wait(5)

win.close()
core.quit()
