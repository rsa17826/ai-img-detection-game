import random


class BalancedRand:
  def __init__(
    self, __minNum=0.0, __maxNum=1.0, __luckImmutability=1.0, __baseLuck=0.5
  ):
    self.baseLuck = __baseLuck
    self.luck = __baseLuck
    self.minNum = __minNum
    self.maxNum = __maxNum
    self.luckImmutability = __luckImmutability

  def next(self):
    distance_to_min = abs(self.luck - 0)
    distance_to_max = abs(self.luck - 1)
    val = min(distance_to_min, distance_to_max)
    rnum = self.clamp(random.gauss(self.luck, val), 0, 1)

    diff = -(rnum - self.baseLuck)
    self.luck += diff * self.luckImmutability
    self.luck = self.clamp(self.luck, 0.0, 1.0)

    return self.rerange(rnum, 0, 1, self.minNum, self.maxNum)

  def clamp(self, value, min_val, max_val):
    return max(min_val, min(value, max_val))

  def rerange(self, value, from_min, from_max, to_min, to_max):
    # Linear interpolation to remap value from one range to another
    return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min
