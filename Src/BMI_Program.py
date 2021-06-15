# Body Mass Index Program

X = int(input('Enter your round weight:'))
result = round(-1.8815 + 0.069223*X)

print('Body Mass Index = {:0.1f}'.format(result))