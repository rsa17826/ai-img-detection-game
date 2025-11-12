from misc import print, f # type:ignore
import itertools
from typing import Any

prettyPrint: bool = True
remainingData: str = ""
UNSET: str = ""


NUMREG = r"(?:nan|inf|-?\d+(?:\.\d+)?)"
SEPREG = r"\s*,\s*"
import random
import string


def randFrom(length, chars):
  return "".join(random.choice(chars) for _ in range(length))


def saveDataToFile(p: str, data) -> None:
  f.write(p, saveData(data).strip())


def loadDataFromFile(p: str, ifUnset=None, progress=None) -> Any:
  d = f.read(p, UNSET)
  if d == UNSET:
    return ifUnset
  d = loadData(d)
  return d if d != UNSET else ifUnset


# fix recursion
def saveData(val, level=0) -> str:
  # print("saveData", val)
  def getIndent(level: int) -> str:
    if not prettyPrint:
      return ""
    indent = "\n" + ("  " * level)
    return indent

  if isinstance(val, int):
    return f"INT({val})"
  elif isinstance(val, float):
    return f"FLOAT({val})"
  elif isinstance(val, str):
    return f"STR({str(val).replace('\\', '\\\\').replace(')', '\\)')})"
  elif isinstance(val, bool):
    return f"BOOL({str(val)})"
  elif val is None: # Represents the TYPE_NIL
    return "NULL()"
  elif isinstance(val, dict): # Assuming dictionary for TYPE_DICTIONARY
    data = ""
    level += 1
    has_key = False
    for inner in val:
      has_key = True
      data += f"{getIndent(level)}{saveData(inner, level)}{saveData(val[inner], level)}"
    level -= 1
    return f"{{{data}{getIndent(level)}}}" if has_key else "{}"
  elif isinstance(val, list): # Assuming list for TYPE_ARRAY
    data = ""
    level += 1
    has_key = False
    for inner in val:
      has_key = True
      data += f"{getIndent(level)}{saveData(inner, level)}"
    level -= 1
    return f"[{data}{getIndent(level)}]" if has_key else "[]"

  print.error(val, type(val))
  return str(val)


def loadData(d: str, progress=None) -> Any:
  global UNSET
  if not UNSET:
    UNSET = ":::" + randFrom(10, "qwertyuiopasdfghjklzxcvbnm1234567890") + ":::"

  remainingData = d.strip() if d else ""
  if not remainingData:
    return UNSET

  def __int(num) -> Any:
    # if num == "inf":
    #   return INF
    # if num == "nan":
    #   return NAN
    return int(num)

  def __float(num) -> float:
    # if num == "inf":
    #   return INF
    # if num == "nan":
    #   return NAN
    return float(num)

  def getDataFind() -> str:
    nonlocal remainingData
    end = remainingData.find(")")
    part = remainingData[1 : end - 1]
    remainingData = remainingData[end + 1 :]
    return part

  _stack: list = []

  while 1:
    if not remainingData:
      # log.warn(_stack, _stack, 4)
      return _stack[len(_stack) - 1]
    if remainingData.startswith("]") or remainingData.startswith("}"):
      remainingData = remainingData[1:] # Remove the first character
      remainingData = remainingData.strip() # Strip whitespace

      # log.pp("DADSADSA", remainingData) # Uncomment if logging is needed

      if not remainingData:
        # log.warn(_stack, _stack, 1) # Uncomment if logging is needed
        # _stack.append(_stack[-1]) # This line is commented out, ensure it's needed.

        while len(_stack) >= 2:
          thing1 = _stack.pop() # pop_back() is equivalent to pop() in Python
          last_item = _stack[-1] # Get the last item

          if last_item is None:
            _stack[-1] = thing1
          elif isinstance(
            last_item, dict
          ): # Check if last_item is a dictionary
            for k in last_item.keys():
              if last_item[k] == UNSET:
                last_item[k] = thing1
                break
          elif isinstance(last_item, list): # Check if last_item is a list
            last_item.append(thing1)

        # log.pp(_stack) # Uncomment if logging is needed
        return _stack[
          0
        ] # Return the first element of the _stack if remainingData is empty

      dataToInsert = _stack.pop() # Equivalent to pop_back()
      thingToPutDataIn = _stack.pop()

      # log.warn(thingToPutDataIn, dataToInsert)
      if isinstance(thingToPutDataIn, dict):
        for k in thingToPutDataIn:
          if thingToPutDataIn[k] == UNSET:
            thingToPutDataIn[k] = dataToInsert
            break
      elif isinstance(thingToPutDataIn, list):
        thingToPutDataIn.append(dataToInsert)

      _stack.append(thingToPutDataIn)
      # remainingData = remainingData
      # _stack.append([remainingData, _stack])
      continue
    remainingData = remainingData.strip()
    if not remainingData:
      print.error(remainingData, "current")
      breakpoint
    t_ype: str = ""
    if remainingData.startswith("{"):
      t_ype = "{"
    elif remainingData.startswith("["):
      t_ype = "["
    else:
      t_ype = remainingData[0 : remainingData.find("(")]
    remainingData = remainingData[len(t_ype) :]
    # if t_ype == UNSET:
    #   log.warn(remainingData)
    remainingData = remainingData.strip()
    # log.pp("asdjhdash", t_ype, remainingData)
    # log.pp(remainingData, t_ype)
    thisdata: Any = None
    if t_ype == "{":
      thisdata = UNSET
      # remainingData = remainingData[1:] # Uncomment if needed for string manipulation
      _stack.append({})

    elif t_ype == "INT":
      thisdata = getDataFind()
      thisdata = __int(thisdata)

    elif t_ype == "FLOAT":
      thisdata = getDataFind()
      thisdata = __float(thisdata)

    elif t_ype == "NULL":
      getDataFind()
      thisdata = None

    elif t_ype == "BOOL":
      thisdata = getDataFind()
      thisdata = thisdata == "True"

    elif t_ype == "STR":
      thisdata = remainingData.replace("\\\\", "ESCAPED" + UNSET).replace(
        r"\)", "PERIN" + UNSET
      )
      thisdata = thisdata[
        1 : thisdata.find(")")
      ] # Adjusting indices for Python slicing
      thisdata = thisdata.replace("ESCAPED" + UNSET, "\\\\").replace(
        "PERIN" + UNSET, ")"
      )
      remainingData = remainingData[
        len(thisdata.replace("\\", "\\\\\\").replace(")", r"\)")) + 2 :
      ]
      thisdata = thisdata.replace("\\\\", "\\")
    elif t_ype == "[":
      thisdata = UNSET
      # remainingData = remainingData[1:] # Uncomment if needed for string manipulation
      _stack.append([])

    else:
      print(f"Error: Bad type - {t_ype} in remainingData: {remainingData}")
      # For debugging purposes, you can use a breakpoint here
      # breakpoint()
      if t_ype == UNSET:
        return UNSET
      return None # Adjust according to the expected behavior

    remainingData = remainingData.strip()
    if thisdata != UNSET:
      if len(_stack):
        lastItem = _stack[len(_stack) - 1]
        if lastItem == None:
          _stack[len(_stack) - 1] = thisdata
        else:
          if isinstance(lastItem, dict):
            innerDataFound = False
            for k in lastItem:
              if lastItem[k] == UNSET:
                innerDataFound = True
                lastItem[k] = thisdata
                break
            if not innerDataFound:
              lastItem[thisdata] = UNSET
          elif isinstance(lastItem, list):
            lastItem.append(thisdata)
      else:
        # _stack.append([remainingData, _stack])
        continue
        # # log.warn("no obj")
        # log.warn(out, _stack, _stack, thisdata, 2)
        # return thisdata
    # log.pp(thisdata, out, remainingData, "_stack:", str(_stack))
    # if len(remainingData):
    # Push the current state back onto the _stack for the next iteration
    # _stack.append([remainingData, _stack])

  # return _stack[len(_stack) - 1]


d: Any = {"asd": 21221, "asdas": [[], {}, [], [21312, {111: 222}]]}
saveDataToFile("./a.sds", d)
print((loadDataFromFile("./a.sds"), d))
print("\n\n\n\r\n\r\n")
