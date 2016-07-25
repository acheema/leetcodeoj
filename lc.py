
import ds
#1
def twosum(nums,target):
    #print twosum([2,7,11,15],22)
    #print "twosum got",nums,"with target",target
    d = dict((v,i) for i,v in enumerate(nums))
    r = set()
    for i,v in enumerate(nums):
        if target-v in d and d[target-v] > i:
            r.add((min(v,target-v),max(v,target-v)))
    return map(list,r)
#2
def addtwonumbers(l1,l2):
    #ds.printlist(addtwonumbers(ds.l1,ds.l2))
    nl,carry = None,0
    while l1 or l2 or carry:
        one = l1.val if l1 else 0
        two = l2.val if l2 else 0
        a = one+two+carry
        newnode = ds.ListNode(a%10)
        carry = 0 if a < 10 else 1
        if not nl:
            nl = newnode
            h = nl
        else:
            nl.next = newnode
            nl = nl.next
        l1 = l1.next if l1 else l1
        l2 = l2.next if l2 else l2
    return h
#3
def longestsubstring(s,k,norepeating = True):
    #longestsubstring("pwwkew",2,True)
    #Also longest repeated substring with at most k chars
    def window_large(norepeating,found,k):
        if norepeating:
            return sum(found.values()) > len(found)
        else:
            return len(found) > k
    found,start,maxstr,L = {},0,-1,0
    while L < len(s):
        found[s[L]] = found[s[L]]+1 if s[L] in found else 1
        while window_large(norepeating,found,k):
            found[s[start]] -= 1
            if found[s[start]] == 0:
                del(found[s[start]])
            start += 1
        if L-start+1 > maxstr:
            maxstart = start
            maxstr = L-start+1
        L += 1
    print s[maxstart:maxstart+maxstr]
    return maxstr
#4
def median(nums1,nums2):
    #print median([1, 12, 15, 26, 38],[2, 13, 17, 30, 45])
    #Do binary search but with medians
    def med(arr):
        n = len(arr)
        if n%2:return arr[n/2]
        return float(arr[n/2]+arr[(n-1)/2])/2
    while len(nums1) != 2 and len(nums2) != 2:
        m1,m2 = med(nums1),med(nums2)
        if m1 == m2:
            return m1
        elif m1 < m2:
            nums1 = [x for x in nums1 if x >= m1]
            nums2 = [x for x in nums2 if x <= m2]
        else:
            nums1 = [x for x in nums1 if x <= m1]
            nums2 = [x for x in nums2 if x >= m2]
    if len(nums1) == len(nums2) == 2:
        return float((max(nums1[0],nums2[0])+min(nums1[1],nums2[1])))/2
#5
def longestpalindrome(s):
    #print longestpalindrome("forgeeksskeegfor")
    i,maxpal,start = 0,1,0
    while i < len(s):
        #Odd
        left,right = i-1,i+1
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right-left+1 > maxpal:
                maxpal,start = right-left+1,left
            left -= 1
            right += 1
        #Even
        left,right = i,i+1
        while left >= 0 and right < len(s) and s[left] == s[right]:
            if right-left+1 > maxpal:
                maxpal,start = right-left+1,left
            left -= 1
            right += 1
        i+=1
    print s[start:start+maxpal]
    return maxpal
#6
def convert(s,numRows):
    #print convert("PAYPALISHIRING",4)
    S,step,row = "",2*numRows-2,0
    while row < numRows:
        if row == 0 or row == numRows-1:
            j = row
            while j < len(s):
                S += s[j]
                j += step
            print "Top or bottom row, finished",S
        else:
            j,f,step1 = row,True,2*(numRows-row-1)
            step2 = step-step1
            while j < len(s):
                print "midrow",j,f,step1,step2,S
                S += s[j]
                j = j + step1 if f else j+step2
                f = not f
        row+=1
    return S
#7
def reverse_integer(x):
    if x > 0:
        if int(str(x)[::-1]) > 2147483648:return 0
        return int(str(x)[::-1])
    return -reverse_integer(str(x)[1:])
#8
def atoi(s):
    #print atoi("    -5678  #$ 4423%^&*")
    def numcat(a,b):
        import math
        if b == 0:return a*10
        return int(math.pow(10,(int(math.log10(b))+ 1))*a + b)
    s = s.lstrip()
    s = s.lstrip("0")
    print "Stripped s",s
    if not s:return 0
    first,negative = 0,False
    if s[0] == "+" or s[0] == "-":
        if s[0] == "-":
            negative = True
        first = 1
    num = 0
    while first < len(s):
        number = ord(s[first])
        print num,first,s[first]
        if number < 48 or number > 57:break
        elif number == 48:num = numcat(num,0)
        elif number == 49:num = numcat(num,1)
        elif number == 50:num = numcat(num,2)
        elif number == 51:num = numcat(num,3)
        elif number == 52:num = numcat(num,4)
        elif number == 53:num = numcat(num,5)
        elif number == 54:num = numcat(num,6)
        elif number == 55:num = numcat(num,7)
        elif number == 56:num = numcat(num,8)
        elif number == 57:num = numcat(num,9)
        else:pass
        first += 1
    if negative:
        if -num < -2147483648:
            return -2147483648
        return -num
    if num > 2147483647:
        return 2147483647
    return num
#9
def isnum_palindrome(x):
    #print isnum_palindrome(12321)
    if x < 0:return False
    og,revd = x,0
    while og:
        revd = revd*10+og%10
        og /= 10
    return revd == x
#10
def ismatch(s, p):
    #print ismatch("aab", "c*a*b")
    m,n = len(s),len(p)
    dp = [[True] + [False] * m]
    for i in xrange(n):
        dp.append([False]*(m+1))
    for i in xrange(1, n + 1):
        x = p[i-1]
        if x == '*' and i > 1:
            dp[i][0] = dp[i-2][0]
        for j in xrange(1, m+1):
            if x == '*':
                dp[i][j] = dp[i-2][j] or dp[i-1][j] or (dp[i-1][j-1] and p[i-2] == s[j-1]) or (dp[i][j-1] and p[i-2]=='.')
            elif x == '.' or x == s[j-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[n][m]
#11
def maxarea(height):
    #print maxarea([1,8,6,2,5,4,8,3,7])
    left,right,maxarea = 0,len(height)-1,-1
    while left != right:
        maxarea = max(maxarea,(right-left)*min(height[left],height[right]))
        if height[left] > height[right]:
            right -= 1
        else:
            left += 1
    return maxarea
#12
def int_to_roman(num):
    #print int_to_roman(3454)
    d = { 3 :  {1 : "M", 2 : "MM", 3 : "MMM"},
          2 :  {1 : "C", 2 : "CC", 3 : "CCC", 4 : "CD", 5 : "D", 6 : "DC", 7 : "DCC", 8 : "DCCC", 9 : "CM"},
          1 :  {1 : "X", 2 : "XX", 3 : "XXX", 4 : "XL", 5 : "L", 6 : "LX", 7 : "LXX", 8 : "LXXX", 9 : "XC"},
          0 :  {1 : "I", 2 : "II", 3 : "III", 4 : "IV", 5 : "V", 6 : "VI", 7 : "VII", 8 : "VIII", 9 : "IX"}
        }
    i, res = 0, ""
    while num:
        dig = num % 10
        if dig != 0:
            res = d[i][dig] +res
        i += 1    
        num /= 10
    return res
#13
def roman_to_int(s):
    #print roman_to_int("MCMXCVI")
    order = {"M":6,"D":5,"C":4,"L":3,"X":2,"V":1,"I":0}
    values = dict([("M",1000),("D",500),("C",100),("L",50),("X",10),("V",5),("I",1)])
    i,result = 0,0
    while i < len(s)-1:
        if order[s[i]] < order[s[i+1]]:
            result += values[s[i+1]]-values[s[i]]
            s = s[:i]+s[i+2:]
            continue
        i += 1
    for l in s:
        result += values[l]
    return result
#14
def longest_common_prefix(strs):
    #print longest_common_prefix(['flower', 'flow', 'floob', 'fleet'])
    if not strs:
        return ""
    letter,longest = 0,0
    while letter < len(max(strs,key=len)):
        print letter,longest
        word = 0
        while word < len(strs)-1:
            word1,word2 = strs[word],strs[word+1]
            word1_len,word2_len = len(word1),len(word2)
            if letter >= word1_len or letter >= word2_len or word1[letter] != word2[letter]:
                return strs[0][:longest]
            word+=1
        longest += 1
        letter += 1
    return strs[0][:longest]
#15
def threesum(nums,target):
    #print threesum([-4,-2,1,-5,-4,-4,4,-2,0,4,0,-2,3,1,-5,0],0)
    nums,r = sorted(nums),set()
    for i in xrange(len(nums)-2):
        left,right = i+1,len(nums)-1
        while left < right:
            s = nums[i]+nums[left]+nums[right]
            if s == target:
                r.add(tuple([nums[i],nums[left],nums[right]]))
                left += 1
                right -= 1
            elif s < target:
                left += 1
            else:
                right -= 1
    return map(list,r)
#16
def threesum_closest(nums,target):
    #print threesum_closest([-1 ,2 ,1 ,-4],1)
    nums,i,result,smallest = sorted(nums),0,0,2**32
    while i < len(nums):
        left,right = i+1,len(nums)-1
        while left < right:
            s = nums[i]+nums[left]+nums[right]
            diff = abs(s-target)
            if diff == 0:
                return s
            if diff < smallest:
                smallest,result = diff,s
            if s > target:
                right -= 1
            else:
                left += 1
        i+=1
    return result
#17
def letter_combinations(digits):
    def combinations(s,letters):
        if len(s) == len(letters):
            r.append(s)
            return
        for letter in letters[len(s)]:
            combinations(s+letter,letters)
    #print letter_combinations("234")
    num_to_letter = {"2":"abc","3":"def","4":"ghi","5":"jkl",
                    "6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
    al = [num_to_letter[l] for l in digits]
    r = []
    combinations("",al)
    return r
#18
def foursum(nums,target):
    r,nums = set(),sorted(nums)
    for i,v in enumerate(nums):
        triplets = threesum(nums[i+1:],target-v)
        if triplets:
            for triplet in triplets:
                r.add(tuple([v,triplet[0],triplet[1],triplet[2]]))
    return map(list,r)
#19
def remove_nth_node(head,n):
    #remove_nth_node(ds.l3,5)
    h,i,restart = head,0,head
    while i < n and h:
        h = h.next
        i += 1
    if not h:
        if i < n:
            return
        head = head.next
        return head
    while h.next:
        restart = restart.next
        h = h.next
    restart.next = restart.next.next
#20
def valid_parenthesis(s):
    #print valid_parenthesis("(({}[[]][]))")
    mapping = {")":"(","]":"[","}":"{"}
    opening,closing = set(["(","{","["]),set(["]","}",")"])
    stack = []
    for char in s:
        if char in opening:
            stack.append(char)
        elif char in closing:
            if not stack or mapping[char] != stack.pop():
                return False
        else:
            raise ValueError("Invalid character")
    return not stack
#21
def merge_sorted_LL(l1,l2):
    #merge_sorted_LL(ds.l1,ds.l2)
    nl,nl_head = None,None
    list1,list2 = l1,l2
    while list1 or list2:
        if not list1:
            if not nl:
                nl,nl_head = list2,list2
            else:
                nl.next = list2
            break
        if not list2:
            if not nl:
                nl,nl_head = list1,list1
            else:
                nl.next = list1
            break
        elif list1.val < list2.val:
            if not nl:
                nl = list1
                nl_head = nl
            else:
                nl.next = list1
                nl = nl.next
            list1 = list1.next
        else:
            if not nl:
                nl = list2
                nl_head = nl
            else:
                nl.next = list2
                nl = nl.next
            list2 = list2.next
    ds.printlist(nl_head)
    return nl_head
#22
def brackets(n):
    #print brackets(3)
    def b(s,opn,close,pairs):
        if opn == pairs and close == pairs:
            r.append(s)
        else:
            if opn < pairs:
                b(s+"(",opn+1,close,pairs)
            if close < opn:
                b(s+")",opn,close+1,pairs)
    r = []
    b("",0,0,n)
    return r
#23
def merge_k_sorted(lists):
    #merge_k_sorted([ds.l1,ds.l2,ds.l3,[]])
    import heapq
    heap,k,nl,nl_head = [],len(lists),None,None
    for i in xrange(k):
        if lists[i]:
            heapq.heappush(heap,(lists[i].val,lists[i]))
    while heap:
        rootval,LL = heapq.heappop(heap)
        if not nl:
            nl = LL
            nl_head = nl
        else:
            nl.next = LL
            nl = nl.next
        if LL.next:
            heapq.heappush(heap,(LL.next.val,LL.next))
    ds.printlist(nl_head)
    return nl_head
#24
def swap_pairs(head):
    #swap_pairs(ds.l1)
    if not head:
        return None
    newh = head.next
    if not newh:
        return head
    a,b,prevb = head,head.next,None
    while a and b:
        nexta = b.next
        a.next = b.next
        b.next = a
        if prevb:
            prevb.next = b
        prevb = a
        a = nexta
        if nexta:
            b = nexta.next
    return newh
#25
def reverse_kgroup(head,k):
    #reverse_kgroup([1,2,3,4,5,6],3) -> [3,2,1,6,5,4]
    #reverse_kgroup([1,2,3,4,5],3) -> [3,2,1,4,5]
    def reverse_LL(head):
        prev,h,nxt = None,head,None
        while h:
            nxt = h.next
            h.next = prev
            prev = h
            h = nxt
        return prev,head
    ds.printlist(head)
    h,newheadsave,prevhead = head,None,None
    while h:
        i = 1
        currhead,curr = h,h
        #Check if the next k elements exist
        while currhead and i < k:
            currhead = currhead.next
            i += 1
        #If they do exist, reverse them
        if currhead and i == k:
            nexthead = currhead.next #save the (k+1)th node
            currhead.next = None #Set the kth.next to None for reversing
            newhead,tail = reverse_LL(curr) #Get the new head and tail
            if not newheadsave: #Save the new head that we will return
                newheadsave = newhead
            tail.next = nexthead #Connect the tail of reversed to (k+1)th node
            if prevhead: #Connect the tail of the last reversed list to the head
                prevhead.next = newhead #of the newly reversed linked list if we've done a reverse
            prevhead = tail #Update so we can connect current reversed list to head of newly reversed one next
            h = nexthead #Start the whole process over again, starting with the (k+1)st node
        #If the k was greater than length of intial list
        #Or the # of remainig elements are less than k
        #Return the appropriate head
        else:
            if newheadsave:
                return newheadsave
            return head
    ds.printlist(newheadsave)
    return newheadsave
#26
def remove_duplicates(nums):
    #print remove_duplicates([1,1,2,3,3,3,5,7])
    i = 0
    while i < len(nums)-1:
        if nums[i] == nums[i+1]:
            del(nums[i+1])
            continue
        i+=1
    return len(nums)
#27
def remove_element(nums,val):
    #print remove_element([3,3,2,2,3],3)
    i = 0
    while i < len(nums):
        if nums[i] == val:
            nums[i] = nums[len(nums)-1]
            nums.pop()
            continue
        i += 1
    return len(nums)
#28
def strStr(needle,haystack):
    #print strStr("needle","jfasdfkasneedlefldksaj")
    def kmp(p,s):
        m,i,T,r = 0,0,kmp_table(p),[]
        while m+i < len(s):
            if p[i] == s[m+i]:
                if i == len(p)-1:
                    return m
                else:
                    i += 1
            else:
                if T[i] > -1:
                    m,i = m+i-T[i],T[i]
                else:
                    m,i = m+1,0
        return -1
    def kmp_table(p):
        T = [0]*len(p)
        T[0],T[1],pos,cnd = -1,0,2,0
        while pos < len(p):
            if p[pos-1] == p[cnd]:
                T[pos] = cnd+1
                cnd += 1
                pos += 1
            elif cnd > 0:
                cnd = T[cnd]
            else:
                T[pos] = 0
                pos+=1
        return T
    return kmp(needle,haystack)
#29
def div(dividend,divisor):
    #print div(121,10)
    if not divisor:
        return sys.maxsize
    neg = (dividend < 0) ^ (divisor < 0)
    result,dividend,divisor = 0,abs(dividend),abs(divisor)
    while dividend >= divisor:
        count = 1
        divisorExp = divisor
        while dividend >= divisorExp:
            dividend -= divisorExp
            result += count
            count += count
            divisorExp += divisorExp
    if neg:
        if result < -2147483648:
            return -2147483648
        else:
            return -result
    else:
        if result > 2147483647:
            return 2147483647
        else:
            return result
#30
def findsubstring(s,words):
    #print findsubstring("barfoofoobarthefoobarman",["bar","foo","the"])
    from collections import Counter
    if not s or not words:
        return []
    r = []
    d = dict(Counter(words)) #Create a occurence dictionary of our target words
    L = len(words[0]) #We will be checking all possible length L substrings
    for j in xrange(L):#Starting from first letter to L'th letter
        cmap = {}
        start,count = j,0
        for i in xrange(j,len(s)-L+1,L):
            substr = s[i:i+L] #Look at every length L substring
            if substr in d: #Check if its a word we want, load into new dict
                cmap[substr] = cmap[substr]+1 if substr in cmap else 1
                count += 1
                """If the frequency of the substr we just loaded is now greater
                than the number we actually need, then slide the window starting from 'start' right -
                going in steps of length L and decreasing the frequency of the left length L substring
                from our window"""
                while cmap[substr] > d[substr]: 
                    left = s[start:start+L]
                    cmap[left] -= 1
                    count -= 1
                    start += L
                """If the dict of our current window matches the dict of the window we want (d)
                then we've found a valid substring that makes up only our target words, so save the
                index of start and then slide the window right by length L, decreasing the frequency
                of the left part of the window by one, in case the input is like so
                s = barfoofoobarthefoobarman, w = ["bar","foo","the"] where the first valid window is
                going to be 'foobarthe' but the next valid window is 'barthefoo' where the 'bar' and
                the 'foo' from the first window are reused in the second window
                """
                if count == len(words):
                    r.append(start)
                    left = s[start:start+L]
                    cmap[left] -= 1
                    count -=1
                    start += L
            else:#Word is not among target words, so our window is broken, try next
                cmap.clear()
                start = i+L #So check the next length L substring
                count = 0 #Clear count
    return r
#31
def next_perm(nums):
    i = len(nums)-1
    while i > 0 and nums[i] <= nums[i-1]:
        i -= 1
    print i
    if i <= 0:
        left,right = 0,len(nums)-1
        while left < right:
            nums[left],nums[right] = nums[right],nums[left]
            left += 1
            right -= 1
        return
    j = len(nums)-1
    while nums[j] <= nums[i-1]:
        j -= 1
    nums[i-1],nums[j] = nums[j],nums[i-1]
    nums[i:] = nums[len(nums) - 1 : i - 1 : -1]
#32
def longest_valid_paren(s):
    #print longest_valid_paren("))((()))")
    result,stack = 0,[]
    for i in xrange(len(s)):
        if s[i] == "(":
            stack.append((i,0))
        else:
            if not stack or stack[len(stack)-1][1] == 1:
                stack.append((i,1))
            else:
                stack.pop()
                curr = i+1 if not stack else i-stack[len(stack)-1][0]
                result = max(result,curr)
    return result
#33
def search_rotate(nums,target):
    #print search_rotate([4,5,6,7,0,1,2],0)
    left,right = 0,len(nums)-1
    while left <= right:
        mid = (left+right)/2
        if nums[mid] == target:
            return mid
        #Left half is already sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target and target < nums[mid]:
                right = mid-1
            else:
                left = mid+1
        #Right half is already sorted
        else:
            if nums[mid] < target and target <= nums[right]:
                left = mid+1
            else:
                right = mid-1
    return -1
#34
def search_range(nums,target):
    #print search_range([5, 7, 7, 8, 8, 10],8)
    def linearScan(start,end,nums,target):
        i = start
        while i < end:
            if nums[i] == target:
                start = i
                while i < end-1 and nums[i] == nums[i+1]:
                    i+=1
                end = i
                return [start,end]
            i+=1
        return [-1,-1]
    mid = len(nums)/2
    if nums[mid] < target: #In the right side
        return linearScan(mid+1,len(nums),nums,target)
    elif nums[mid] > target: #In the left side
        return linearScan(0,mid,nums,target)
    elif nums[mid] == target:#Currently in the window, expand both directions
        left,right = mid,mid
        while left >= 0 and nums[left] == target:
            left -=1
        while right < len(nums) and nums[right] == target:
            right += 1
        return [left+1,right-1]
    else:
        return [-1,-1]
#35
def search_insert(nums,target):
    #print search_insert([1,3,5,7,9,10],8)
    left,right = 0,len(nums)
    while left < right:
        print left,right
        mid = (left+right)/2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid+1
        else:
            right = mid
    if left < len(nums) and nums[left] < target:
        return left+1
    return left
#36
def valid_suduko(board):
    """b=["534678912",
        "672195348",
        "198342567",
        "859761423",
        "426853791",
        "713924856",
        "961537284",
        "287419635",
        "345286179"]"""
    for row in board:
        d = {}
        for val in row:
            if val != ".":
                d[val] = d.get(val,0)+1
        if sum(d.values()) > len(d):
            return False
    for column in xrange(len(board[0])):
        d = {}
        for rows in xrange(len(board)):
            val = board[rows][column]
            if val != ".":
                d[val] = d.get(val,0)+1
        if sum(d.values()) > len(d):
            return False
    #Check boxes
    for row in xrange(0,len(board),3): #Row 0,3,6
        for column in xrange(0,len(board[0]),3): #Column 0,3,6
            d = {}
            for blockrow in board[row:row+3]: #Grab all rows
                for item in blockrow[column:column+3]: #Get only items in the box
                    if item != ".":
                        d[item] = d.get(item,0)+1
            for val in d.values():
                if val > 1:
                    return False
    return True
#37
def solve_sudoku(board):
    #board = ["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
    #solve_sudoku(board)
    def sudoku(board,setrow,setcol,setval): #Bulk of the solver, recursive heavy lifter
        board[setrow][setcol] = setval #Set the row and col to setval
        if valid_and_full(board): #Check if we're done
            return True
        for r in xrange(9): #Find the next possible slot
            for c in xrange(9):
                if board[r][c] == ".":
                    nums = valid_nums(board,r,c) #See if we have any possible moves
                    for num in nums:
                        if sudoku(board,r,c,num):
                            return True #If we found a valid board, propagate
                    """If no valid moves found, or all the valid moves we tried led
                    us to dead ends where everything was invalid, remove the value
                    we just put and return False, so the stack above can try
                    another possible move or if none exist, do the same for the stack
                    above it"""
                    board[setrow][setcol] = "."
                    return False
    def valid_and_full(board):
        for row in board:
            for item in row:
                if item == ".":
                    return False
        return valid_suduko(board)
    def valid_nums(board,i,j):
        valid = set(map(str,set(xrange(1,10)))) #Initially everything is possible
        row = board[i]
        for item in row: #Remove all vals in row
            if item in valid:
                valid.remove(item)
        column = []
        for r in board:
            column.append(r[j])
        for item in column:#Remove all vals in col
            if item in valid:
                valid.remove(item)
        if i < 3:
            startrow = 0
        elif i < 6:
            startrow = 3
        else:
            startrow = 6
        if j < 3:
            startcol = 0
        elif j < 6:
            startcol = 3
        else:
            startcol = 6
        #Remove all values seen in box
        for row in board[startrow:startrow+3]:
            for item in row[startcol:startcol+3]:
                if item in valid:
                    valid.remove(item)
        return list(valid)
    
    #High level intialization function
    board = map(list,board)
    for i,row in enumerate(board):
        for j,item in enumerate(row):
            if item == ".": #Find the first slot and start there
                for num in valid_nums(board,i,j):
                    if sudoku(board,i,j,num):
                        break #If we found a valid one, we are done
    for t in board: #Actual board is modified in place, but we want to see solution
        print t
#38
def count_and_say(n):
    #print count_and_say(5)
    def get_next(nums):
        i,newnum = 0,[]
        while i < len(nums):
            count = 1
            while i < len(nums)-1 and nums[i] == nums[i+1]:
                count += 1
                i+=1
            newnum.append(count)
            newnum.append(nums[i])
            i+=1
        return newnum
    curr = [1]
    for i in xrange(1,n):
        print "In func",curr
        curr = get_next(curr)
    return "".join(map(str,curr))
#39
def uniquecoins(coins,value):
    #print uniquecoins([8,7,4,3],11)
    def recurse(coins,value,j,curr,result):
        if value == 0:
            result.append(curr[:])
            return
        for i in xrange(j,len(coins)):
            if value < coins[i]:
                return
            curr.append(coins[i])
            recurse(coins,value-coins[i],i,curr,result)
            curr.pop()
    result = [[]]
    recurse(sorted(coins),value,0,[],result)
    return filter(lambda x: x,result)
#40
def uniquecoins_nodupes(candidates,target):
    #print uniquecoins_nodupes([10, 1, 2, 7, 6, 1, 5],8)
    def recurse(coins,value,j,curr,result):
        if value == 0:
            result.add(tuple(curr[:]))
            return
        for i,v in enumerate(coins[j:]):
            if value < v:
                return
            curr.append(v)
            recurse([y for x,y in enumerate(coins[j:]) if x != i],value-v,i,curr,result)
            curr.pop()
    result = set()
    recurse(sorted(candidates),target,0,[],result)
    return map(list,result)
#41
def first_missing_positive(nums):
    #print first_missing_positive([3,4,-1,1])
    nums = [v for v in nums if v > 0]
    for i,v in enumerate(nums):
        if abs(v)-1 < len(nums) and nums[abs(v)-1] > 0:
            nums[abs(v)-1] *= -1
    for i,v in enumerate(nums):
        if v > 0:
            return i+1
    return len(nums)+1
#42
def trap_rainwater(height):
    #print trap_rainwater([0,1,0,2,1,0,1,3,2,1,2,1])
    #135
    def candy(ratings):
        #print candy([2,4,2,6,1,7,8,9,2,1])
        candies = [1]+[0]*(len(ratings)-1)
        for i in xrange(1,len(ratings)):
            candies[i] = candies[i-1]+1 if ratings[i] > ratings[i-1] else 1
        min_needed = candies[len(candies)-1]
        for j in reversed(xrange(len(ratings)-1)):
            curr = candies[j+1]+1 if ratings[j] > ratings[j+1] else 1
            min_needed += max(curr,candies[j])
            candies[j] = curr
        return min_needed
    left,right = [-1]*len(height),[-1]*len(height)
    maxh,left[0] = height[0],height[0]
    for i in xrange(1,len(height)):
        if height[i] < maxh:
            left[i] = maxh
        else:
            left[i] = height[i]
            maxh = height[i]
    right[len(height)-1] = height[len(height)-1]
    maxh = height[len(height)-1]
    for i in reversed(xrange(len(height)-1)):
        if height[i] < maxh:
            right[i] = maxh
        else:
            right[i] = height[i]
            maxh = height[i]
    result = 0
    for i in xrange(len(height)):
        result += min(left[i],right[i])-height[i]
    return result
#43
def multiply_strs(num1,num2):
    #print multiply_strs("123","45")
    result = [0]*(len(num1)+len(num2))
    for i in reversed(xrange(len(num1))):
        for j in reversed(xrange(len(num2))):
            val = (ord(num1[i]) - ord("0"))*(ord(num2[j]) - ord("0"))
            p1,p2 = i+j,i+j+1
            sm = val+result[p2]
            result[p1] += sm/10 #Carry
            result[p2] = sm%10 #Remainder
    ret = ("".join(map(str,result))).lstrip("0")
    if not ret:
        return "0"
    return ret
#44
def wildcard_match(s,p):
    x = 1
    """COME BACK TO THIS
    LATER"""
#45
def jump_game(nums):
    #print jump_game([6,2,6,1,7,9,3,5,3,7,2,8,9,4,7,7,2,2,8,4,6,6,1,3])
    if len(nums) <= 1:
        return 0
    start,end = 0,nums[0]
    step = 1
    max_dist = nums[0]
    #Basically does a BFS, finding the max reachable from our current window
    while end < len(nums)-1:
        for i in xrange(start+1,end+1):#In our current window, see if we've found a bigger end val
            max_dist = max(max_dist,i+nums[i])
        start = end #New window starts where our old one ended
        end = max_dist #New end is the max end we found in our prev window
        step+=1 #Everything in that last window was reachable in 'step' jumps, so new window is reachable in step+1
    return step
#46
def permute(nums):
    #print permute([1,2,3])
    if len(nums) == 1:
        return [nums]
    r = []
    for i,v in enumerate(nums):
        r += [[v]+p for p in permute(nums[:i]+nums[i+1:])]
    return r
#47
def permuteUnique(nums):
    #print permuteUnique([1,2,1])
    def combinations(nums,path,res):
        print nums,path,res
        if not nums:
            res.append(path)
            return
        i,n = 0,len(nums)
        for i in xrange(n):
            if 0 < i < n and nums[i] == nums[i-1]: #This would lead to duplicates
                continue
            if i < n:
                curr = nums.pop(i)
                combinations(nums,path+[curr],res)
                nums.insert(i,curr)
    path,res = [],[]
    combinations(sorted(nums),path,res)
    return res
#48
def rotate_img(matrix):
    #rotate_img([["A","B","C","D"],["E","F","G","H"],["I","J","K","L"],["M","N","O","P"]])
    def rotate_layer(matrix,layer):
        first = layer
        last = len(matrix)-1-layer
        for i in xrange(first,last):
            offset = i-first
            top = matrix[first][i]
            right = matrix[i][last]
            bottom = matrix[last][last-offset]
            left = matrix[last-offset][first]
            #Swap them
            temp = top
            matrix[first][i] = matrix[last-offset][first] #Top is now left
            matrix[last-offset][first] = matrix[last][last-offset] #Left is now bottom
            matrix[last][last-offset] = matrix[i][last] #Bottom is now right
            matrix[i][last] = temp #Right is now temp
    for layer in xrange(len(matrix)/2):
        rotate_layer(matrix,layer)
#49
def group_anagrams(strs):
    #print group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) 
    hashtable = {}
    for string in strs:
        sort = tuple(sorted(string)) #Can also use our own ascii bucket sort
        hashtable[sort] = hashtable.get(sort,[])+[string]
    return list(hashtable.values())
#50
def pow(x,n):
    #print pow(3,7)
    if n < 0:
        return pow(1.0/x,-n)
    if n == 0:
        return 1
    if n == 1:
        return x
    if n%2: #We decrease by half because thats is essentially taking two terms and group
        return x*pow(x*x,(n-1)/2)
    return pow(x*x,n/2) #Ex (3*3*3*3*3*3) = (9*9*9) = 9^3





















