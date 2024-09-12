import numpy as np



class Mapper:
    def __init__(self, buttons, main_stick, trigger):
        self.buttons = buttons # [A, B, X(dont use), Y, Z]
        self.main_stick = main_stick # (x, y)
        self.trigger = trigger # (L, R)

        self.boolean_part = np.array([False, False, False, False, False, False, False], dtype=bool)
        self.float_part = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        
    def get_controller_state(self):
        # Z > X=Y > A > B > L=R
        for i in (4, 3, 0, 1):
            if self.buttons[i] > 0.5:
                self.boolean_part[i] = True
                break
        if (True not in self.boolean_part) and self.trigger[0] > 0.2:
            self.boolean_part[5] = True

        np.argmax(self.main_stick)
        self.float_part = StickSelector.closest_stick(, self.main_stick)
        
        self.controller_state = self.boolean_part.tolist() + self.float_part.tolist()
        return self.controller_state



class StickSelector:
    def __init__(self):
        self.no_stick = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_Y_stick = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_Y_stick = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_X_stick = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_X_stick = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_X_neg_Y_stick = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_X_pos_Y_stick = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_X_neg_Y_stick = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_X_pos_Y_stick = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_pos_X_stick = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_neg_X_stick = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_pos_Y_stick = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_neg_Y_stick = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)

        self.sticks = [
            self.no_stick, self.neg_Y_stick, self.pos_Y_stick, 
            self.neg_X_stick, self.pos_X_stick, self.neg_X_neg_Y_stick, 
            self.neg_X_pos_Y_stick, self.pos_X_neg_Y_stick, self.pos_X_pos_Y_stick, 
            self.tilt_pos_X_stick, self.tilt_neg_X_stick, self.tilt_pos_Y_stick, 
            self.tilt_neg_Y_stick
        ]
    
    def adjust_for_diagonal(self, stick):
        if np.abs(stick[0]) == 1.0 and np.abs(stick[1]) == 1.0:
            return np.array([stick[0] / np.sqrt(2), stick[1] / np.sqrt(2)])
        return stick[:2]
    
    def closest_stick(self, x, y):
        stick_2d_adjusted = np.array([self.adjust_for_diagonal(stick) for stick in self.sticks])
        distances = np.linalg.norm(stick_2d_adjusted - np.array([x, y]), axis=1)
        closest_index = np.argmin(distances)
        return self.sticks[closest_index]

class InverseMapper:
    def __init__(self):
        self.no_button = np.array([False, False, False, False, False, False, False], dtype=bool)
        self.l_button = np.array([False, False, False, False, False, True, False], dtype=bool)
        self.a_button = np.array([True, False, False, False, False, False, False], dtype=bool)
        self.b_button = np.array([False, True, False, False, False, False, False], dtype=bool)
        self.y_button = np.array([False, False, False, True, False, False, False], dtype=bool)     
        self.z_button = np.array([False, False, False, False, True, False, False], dtype=bool)
        
        self.no_stick = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_Y_stick = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_Y_stick = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_X_stick = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_X_stick = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_X_neg_Y_stick = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.neg_X_pos_Y_stick = np.array([-1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_X_neg_Y_stick = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.pos_X_pos_Y_stick = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_pos_X_stick = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_neg_X_stick = np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_pos_Y_stick = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.tilt_neg_Y_stick = np.array([0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.selector = StickSelector()

    # Z > X=Y > A > B > L=R    
    def __call__(self, controller_state):
        boolean_part = np.array(controller_state[:7], dtype=bool)
        float_part = np.array(controller_state[7:], dtype=float)
        x, y = float_part[:2]
        float_part = selector.closest_stick(x, y)
        
        if np.array_equal(boolean_part, self.no_button):
            if np.array_equal(float_part, self.no_stick):
                return 0
            if np.array_equal(float_part, self.neg_Y_stick):
                return 1
            if np.array_equal(float_part, self.pos_Y_stick):
                return 2
            if np.array_equal(float_part, self.neg_X_stick):
                return 3
            if np.array_equal(float_part, self.neg_X_neg_Y_stick):
                return 4
            if np.array_equal(float_part, self.neg_X_pos_Y_stick):
                return 5
            if np.array_equal(float_part, self.pos_X_stick):
                return 6
            if np.array_equal(float_part, self.pos_X_neg_Y_stick):
                return 7
            if np.array_equal(float_part, self.pos_X_pos_Y_stick):
                return 8
            
        if boolean_part[4]:#z
            return 33
        if boolean_part[3] or boolean_part[2]:#y, x
            return 32
        if boolean_part[0]: # a
            if np.array_equal(float_part, self.neg_X_stick):
                return 18
            if np.array_equal(float_part, self.pos_X_stick):
                return 19
            if np.array_equal(float_part, self.neg_Y_stick):
                return 20
            if np.array_equal(float_part, self.pos_Y_stick):
                return 21
            if np.array_equal(float_part, self.no_stick):
                return 22
            if np.array_equal(float_part, self.tilt_neg_X_stick):
                return 28
            if np.array_equal(float_part, self.tilt_pos_X_stick):
                return 29
            if np.array_equal(float_part, self.tilt_neg_Y_stick):
                return 30
            if np.array_equal(float_part, self.tilt_pos_Y_stick):
                return 31
        if boolean_part[1]:#b
            if np.array_equal(float_part, self.neg_X_stick):
                return 23
            if np.array_equal(float_part, self.pos_X_stick):
                return 24
            if np.array_equal(float_part, self.neg_Y_stick):
                return 25
            if np.array_equal(float_part, self.pos_Y_stick):
                return 26
            if np.array_equal(float_part, self.no_stick):
                return 27
            
        if boolean_part[5] or boolean_part[6]:
            if np.array_equal(float_part, self.no_stick):
                return 9
            if np.array_equal(float_part, self.neg_Y_stick):
                return 10
            if np.array_equal(float_part, self.pos_Y_stick):
                return 11
            if np.array_equal(float_part, self.neg_X_stick):
                return 12
            if np.array_equal(float_part, self.neg_X_neg_Y_stick):
                return 13
            if np.array_equal(float_part, self.neg_X_pos_Y_stick):
                return 14
            if np.array_equal(float_part, self.pos_X_stick):
                return 15
            if np.array_equal(float_part, self.pos_X_neg_Y_stick):
                return 16
            if np.array_equal(float_part, self.pos_X_pos_Y_stick):
                return 17
        return 0
