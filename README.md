# 챗봇 구현
# 레벤츄타인 거리를 이용한 챗봇 구하기

    import pandas as pd
    import numpy as np

# 챗봇 클래스를 정의
    class SimpleChatBot:
    
# 챗봇 객체를 초기화하는 메서드, 초기화 시에는 입력된 데이터 파일을 로드하고, 질문 데이터를 저장함
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)

   # CSV 파일로부터 질문과 답변 데이터를 불러오는 메서드
    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()
        answers = data['A'].tolist()
        return questions, answers

   # 입력 문장에 가장 잘 맞는 답변을 찾는 메서드, 입력 문장과 기존 질문 간의 Levenshtein distance를 계산하여 가장 작은 거리를 가진 질문의 답변을 반환함
    def find_best_answer(self, input_sentence):
        distances = []
        for question in self.questions:
            distance = self.levenshtein_distance(input_sentence, question)
            distances.append(distance)

        min_distance = min(distances)
        index = distances.index(min_distance)
        return self.answers[index]

   # Levenshtein distance를 계산하는 메서드
    def levenshtein_distance(self, s1, s2):
        m = len(s1)
        n = len(s2)
        dp = np.zeros((m+1, n+1))
       
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
       
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
       
        return dp[m][n]

   # 데이터 파일의 경로를 지정합니다.
    filepath = 'ChatbotData.csv'

   # 챗봇 객체를 생성합니다.
    chatbot = SimpleChatBot(filepath)

   # '종료'라는 입력이 나올 때까지 사용자의 입력에 따라 챗봇의 응답을 출력하는 무한 루프를 실행합니다.
    while True:
        input_sentence = input('You: ')
       if input_sentence.lower() == '종료':
          break
      response = chatbot.find_best_answer(input_sentence)
      print('Chatbot:', response)
