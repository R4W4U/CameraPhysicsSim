import cv2
import threading
import time
import numpy as np

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_BUFFERSIZE,1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

latest_frame = None
running = True
WORLD_TOP = 100

def camera_thread():
    global latest_frame
    while running:
        ret,frame = cam.read()
        if ret:
            latest_frame = frame

thread = threading.Thread(target=camera_thread)
thread.start()

class RectangleObject:
    def __init__(self,contour):
        rect = cv2.minAreaRect(contour)
        self.points = cv2.boxPoints(rect)
        self.points = self.points.astype(float)
        self.points[:,1] -= WORLD_TOP

    def draw(self,img):
        pts = self.points.copy()
        pts[:,1] += WORLD_TOP
        pts = pts.astype(int)
        cv2.drawContours(
            img,
            [pts],
            -1,
            (0,255,0),
            3
        )

    def collision(self,x,y,radius):
        closest = None
        normal = None
        minimum = 999999

        for i in range(4):
            p1 = self.points[i]
            p2 = self.points[(i+1)%4]

            edge = p2-p1
            length = edge.dot(edge)

            if length == 0:
                continue

            t = (
                (x-p1[0])*edge[0]+
                (y-p1[1])*edge[1]
            ) / length

            t = max(0,min(1,t))

            point = p1+t*edge

            dx = x-point[0]
            dy = y-point[1]

            distance = (dx*dx+dy*dy)**0.5

            if distance < minimum:
                minimum = distance
                closest = point

                edge_length = (
                    edge[0]**2+
                    edge[1]**2
                )**0.5

                if edge_length:
                    normal = [
                        -edge[1]/edge_length,
                        edge[0]/edge_length
                    ]

        if closest is not None and minimum < radius:
            direction = [
                x-closest[0],
                y-closest[1]
            ]

            if direction[0]*normal[0]+direction[1]*normal[1] < 0:
                normal[0] *= -1
                normal[1] *= -1

            return normal

        return None

class Ball:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = 12
        self.gravity = 700

    def update(self,dt,objects,width,height):
        self.vy += self.gravity*dt
        self.x += self.vx*dt
        self.y += self.vy*dt

        if self.x-self.radius < 0:
            self.x = self.radius
            self.vx *= -0.8

        if self.x+self.radius > width:
            self.x = width-self.radius
            self.vx *= -0.8

        for obj in objects:
            normal = obj.collision(
                self.x,
                self.y,
                self.radius
            )

            if normal:
                velocity = (
                    self.vx*normal[0]+
                    self.vy*normal[1]
                )

                if velocity < 0:
                    self.vx -= 2*velocity*normal[0]
                    self.vy -= 2*velocity*normal[1]

                    self.vx *= 0.85
                    self.vy *= 0.85

                    self.x += normal[0]*5
                    self.y += normal[1]*5

    def draw(self,img):
        y = self.y + WORLD_TOP

        if -self.radius < y < img.shape[0]+self.radius:
            cv2.circle(
                img,
                (int(self.x),int(y)),
                self.radius,
                (0,0,255),
                -1
            )

    def outside(self,height):
        return self.y-self.radius > height

def create_ball(width):
    return Ball(
        width//2,
        -WORLD_TOP
    )

balls = []
objects = []
edges_cache = None

last_detection = time.time()
detection_interval = 0.5

last_spawn = time.time()
spawn_interval = 1.0

last_time = time.time()

#for ball window
cv2.namedWindow(
    "Ball only",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Ball only",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    now = time.time()

    if latest_frame is None:
        continue

    frame = latest_frame.copy()

    if now-last_detection > detection_interval:
        last_detection = now

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (9,9),
            0
        )

        threshold = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            7
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (7,7)
        )

        clean = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        clean = cv2.morphologyEx(
            clean,
            cv2.MORPH_CLOSE,
            kernel
        )

        edges_cache = clean.copy()

        contours,_ = cv2.findContours(
            clean,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []

        frame_area = frame.shape[0]*frame.shape[1]

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 3000:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                0.03*perimeter,
                True
            )

            if len(approx) != 4:
                continue

            if not cv2.isContourConvex(approx):
                continue

            rect = cv2.minAreaRect(contour)

            w,h = rect[1]

            if w == 0 or h == 0:
                continue

            ratio = max(w,h)/min(w,h)

            if ratio > 3:
                continue

            if area > frame_area/10:
                continue

            objects.append(
                RectangleObject(contour)
            )

    dt = now-last_time
    last_time = now

    if dt > 0.05:
        dt = 0.05

    height,width = frame.shape[:2]
    world_height = height + WORLD_TOP

    if now-last_spawn > spawn_interval:
        last_spawn = now
        balls.append(
            create_ball(width)
        )

    for ball in balls[:]:
        ball.update(
            dt,
            objects,
            width,
            world_height
        )

        if ball.outside(world_height):
            balls.remove(ball)

    output = frame.copy()

    # Empty black frame for only the balls
    ball_frame = np.full(
        frame.shape,
        255,
        dtype=np.uint8
    )

    for obj in objects:
        obj.draw(output)

    for ball in balls:
        ball.draw(output)
        ball.draw(ball_frame)

    cv2.imshow(
        "Physics World",
        output
    )

    cv2.imshow(
        "Ball only",
        ball_frame
    )

    if edges_cache is not None:
        cv2.imshow(
            "Edges",
            edges_cache
        )

    if cv2.waitKey(1)==27:
        break

running = False
thread.join()
cam.release()
cv2.destroyAllWindows()
