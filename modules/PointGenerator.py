import numpy as np

class PointGenerator:
    def __init__(self, data_num=2000):
        self.data_num = data_num
    
    def generate_box(self):
        xx = np.random.uniform(0.3, 0.7, self.data_num) + 0.012 * np.random.randn(self.data_num)  # x 좌표 계산
        yy = np.random.uniform(0.3, 0.7, self.data_num) + 0.012 * np.random.randn(self.data_num)  # y 좌표 계산
        return self._combine_and_shuffle(xx, yy)

    
    def generate_2box(self):
        # 중심이 (0.4, 0.4), 한 변 크기 0.4인 정사각형, 중심이 (0.6, 0.6), 한 변 크기 0.4인 정사각형이 겹친 영역
        # 이 영역 내에서 uniform하게 데이터 생성
        # [0, 0.1] * [0, 0.1] 범위에서 랜덤하게 점 추출하고, 해당 영역 안에 있으면 점 추가, self.data_num 쌓일 때 까지 반복
        ret = []
        while len(ret) < self.data_num:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            if (abs(x-0.4)<=0.2 and abs(y-0.4)<=0.2) or (abs(x-0.6)<=0.2 and abs(y-0.6)<=0.2):
                ret.append((x, y))
        ret = np.array(ret)
        return ret
    
    def generate_biased_diamond(self):
        # 중심과 팔 길이 설정
        center_x, center_y = 0.6, 0.6
        arm_length = 0.2
        
        # 기본 사각형 좌표 생성 (x, y) 범위는 (0.4, 0.8)로 설정
        xx = np.random.uniform(center_x - arm_length, center_x + arm_length, self.data_num)
        yy = np.random.uniform(center_y - arm_length, center_y + arm_length, self.data_num)
        
        # 다이아몬드 분포로 만들기 위해 45도 회전 변환을 적용
        # 회전 변환 행렬: [[cos(45도), -sin(45도)], [sin(45도), cos(45도)]]
        theta = np.radians(45)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta),  np.cos(theta)]])
        
        # 좌표를 중심 기준으로 변환하기 위해 중심에서 빼고 회전 후 다시 더함
        coords = np.vstack((xx - center_x, yy - center_y))  # 중심 이동
        rotated_coords = np.dot(rotation_matrix, coords)    # 45도 회전
        xx_rotated, yy_rotated = rotated_coords[0] + center_x, rotated_coords[1] + center_y  # 다시 중심으로 복원
        
        # 남은 데이터를 셔플해서 반환
        return self._combine_and_shuffle(xx_rotated, yy_rotated)

    def generate_I(self):
        xlist, ylist = [], []
        while len(xlist) < self.data_num:
            x, y = np.random.uniform(0, 1), np.random.uniform(0, 1)
            # 사각형 1, 2, 3 중 하나라도 포함되면 조건 만족
            if ((0.52 <= x <= 0.68) and (0.65 <= y <= 0.7)) or \
            ((0.52 <= x <= 0.68) and (0.5 <= y <= 0.55)) or \
            ((0.57 <= x <= 0.63) and (0.5 <= y <= 0.7)):
                xlist.append(x)
                ylist.append(y)
        return self._combine_and_shuffle(np.array(xlist), np.array(ylist))
    
    def generate_heart(self):
        xlist = []
        ylist = []
        # self.data_num 만큼의 점을 생성합니다.
        a, b, c, d, k = 0.5, 0.2, 0.6, 0.15, 0.53
        while len(xlist) < self.data_num:
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            xx = (x-c)/b
            yy = (y-k)/d
            if xx**2 + (yy - np.sqrt(np.abs(xx)))**2 <= a:
                xlist.append(x)
                ylist.append(y)
        xlist = np.array(xlist)
        ylist = np.array(ylist)
        # 내부적으로 점들을 합치고 섞는 함수 사용 (예: self._combine_and_shuffle)
        return self._combine_and_shuffle(xlist, ylist)

    def generate_Q(self):
        xlist, ylist = [], []
        while len(xlist) < self.data_num:
            x, y = np.random.uniform(0, 1), np.random.uniform(0, 1)
            # 사각형 1, 2, 3 중 하나라도 포함되면 조건 만족
            rr = (x-0.6)**2 + (y-0.6)**2
            xy = x+y
            yx = y-x
            if (0.005 <= rr <= 0.015) or ((1.17 <= xy <= 1.23) and (-0.23 <= yx <= -0.07)):
                xlist.append(x)
                ylist.append(y)
        return self._combine_and_shuffle(np.array(xlist), np.array(ylist))
    
    def _combine_and_shuffle(self, xx, yy):
        data = np.column_stack((xx, yy))  # x와 y 좌표를 합쳐서 데이터 생성
        np.random.shuffle(data)
        return data