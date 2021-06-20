import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode = False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(  self.mode, self.maxHands, self.detectionCon, self.trackCon )
        self.mpdraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:

                    self.mpdraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw= True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmlist.append([id, cx, cy])


        return lmlist



def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == (27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
