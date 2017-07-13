import copy, time
import random

def permute(depth):
	return permuteHelper([], depth, 1, [])


def permuteHelper(a, depth, num, result):
	if num == depth + 1:
		result.append(a)
	else:
		for index in range(num):
			b = copy.copy(a)
			b.insert(index, num)
			permuteHelper(b, depth, num + 1, result)
	return result

class Obj(object):
	def __init__(self):
		self.x = random.random()
	def __repr__(self):
		return str(self.x)


a = [Obj() for i in range(10)]
b = sorted(a, key = lambda o: o.x, reverse = True)[:3]
print (b)