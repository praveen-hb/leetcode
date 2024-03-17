import java.util.*;

public class Practice {
    public ListNode removeZeroSumSublists(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode dummy = new ListNode(0, head);
        ListNode cur = head;
        int prefixSum = 0;
        Map<Integer, ListNode> hash = new HashMap<>();
        hash.put(0, dummy);
        while (cur != null) {
            prefixSum += cur.val;
            if (hash.containsKey(prefixSum)) {
                ListNode prev = hash.get(prefixSum);
                ListNode temp = prev.next;
                int cSum = prefixSum + temp.val;
                while (temp != cur) {
                    temp = temp.next;
                    hash.remove(cSum);
                    cSum += temp.val;
                }
                prev.next = cur.next;
            } else {
                hash.put(prefixSum, cur);
            }
            cur = cur.next;
        }
        return dummy.next;

    }

    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        if (k == 0) return;
        int count = 0;
        for (int start = 0; count < nums.length; start++) {
            int next = start;
            int prev = nums[start];
            do {
                next = (next + k) % nums.length;
                int temp = nums[next];
                nums[next] = prev;
                prev = temp;
                count++;
            } while (next != start);
        }
    }


    public boolean isValidSudoku(char[][] board) {
        for (int i = 0; i < 9; i += 3) {
            for (int j = 0; j < 9; j += 3) {
                Set<Character> set = new HashSet<>();
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        if (board[i + a][j + b] != '.') {
                            if (set.contains(board[i + a][j + b])) return false;
                            set.add(board[i + a][j + b]);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < 9; i++) {
            Set<Character> setR = new HashSet<>();
            Set<Character> setC = new HashSet<>();
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    if (setR.contains(board[i][j])) return false;
                    setR.add(board[i][j]);
                }
                if (board[j][i] != '.') {
                    if (setC.contains(board[j][i])) return false;
                    setC.add(board[j][i]);
                }
            }
        }
        return true;
    }



    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        int sign = 1;
        int result = 0;
        int operand = 0;
        for(char c : s.toCharArray()){
            if(Character.isDigit(c)){
                operand = operand*10 + (c - '0');
            } else if(c!=' '){
                if(c=='+'){
                    result+=sign*operand;
                    sign =1;
                    operand = 0;
                } else if(c=='-'){
                    result+=sign*operand;
                    sign = -1;
                    operand = 0;
                } else if(c=='('){
                    stack.push(result);
                    stack.push(sign);
                    sign = 1;
                    operand = 0;
                    result = 0;
                } else if(c==')'){
                   result += sign * operand;
                   result = stack.pop()*result + stack.pop();
                   operand = 0;
                   sign = 1;
                }
            }
        }
        return result + sign*operand;
    }

    public int eval(Stack<Object> stack){
        if(!stack.isEmpty()) return 0;
        if(stack.peek() instanceof Integer){
            stack.push(0);
        }
        int result = (int) stack.pop();
        while(!stack.isEmpty() && Objects.equals(stack.peek(),')')){
            char operator = (char) stack.pop();
            int num = (int) stack.pop();
            if(operator =='+'){
                result+=num;
            } else if(operator=='-'){
                result-=num;
            }
        }
        stack.pop();
        return result;
    }

    public int calculate2(String s) {
        Stack<Object> stack = new Stack<>();
        int operand = 0;
        int n= 0;

        for(int i=s.length()-1;i>=0;i++){
            char c= s.charAt(i);
            if(Character.isDigit(c)){
                operand = (int) Math.pow(10,n) * (c-'0') + operand;
                n++;
            } else if(c!=' '){
                if(c=='+' || c=='-'){
                    stack.push(operand);
                    operand = 0;
                    n =0;
                    stack.push(c);
                } else if(c==')'){
                    stack.push(c);
                } else{
                    stack.push(eval(stack));
                }
            }
        }
        return eval(stack);
    }



    public int pivotInteger(int n) {
        int lo = 1, hi = n;
        int sumOfN = n * (n+1)/2;
        while(lo<=hi){
            int mid = lo + (hi-lo)/2;
            int leftSum = (mid*(mid+1))/2;
            int rightSum = sumOfN - leftSum;
            if(leftSum==rightSum) return mid;
            else if(leftSum<rightSum){
                lo = mid+1;
            } else{
                hi = mid-1;
            }
        }
        return -1;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode listNode = new ListNode(0);
        ListNode head = listNode;
        int c =0;
        while(l1!=null || l2!=null){
          if(l1!=null){
              c += l1.val;
              l1 = l1.next;
          }
          if(l2!=null){
              c += l2.val;
              l2 = l2.next;
          }
          listNode.next = new ListNode(c/10);
          listNode = listNode.next;
          c = c%10;
        }
        if(c>0){
            listNode.next  = new ListNode(c);
        }
        return head.next;
    }

    public int candy(int[] ratings) {
        int n= ratings.length;
        int[] candies = new int[n];
        Arrays.fill(candies, 1);
        for(int i=1;i<n;i++){
            if(ratings[i]>ratings[i-1]){
                candies[i] = candies[i-1]+1;
            }
        }
        int totalCandies = candies[n-1];
        for(int i=n-2;i>=0;i--){
            if(ratings[i]>ratings[i-1] && candies[i]<=candies[i-1]){
                candies[i] = candies[i-1] + 1;
            }
            totalCandies+=candies[i];
        }
        return totalCandies;
    }

    public int sum(int n){
        return (n * (n+1))/2;
    }

    public int candy2(int[] ratings) {
        int up = 0, down = 0, candies =0, newSlope = 0, oldSlope = 0;
        for(int i=1;i<ratings.length; i++){
            if(ratings[i]>ratings[i-1]) newSlope = 1;
            else if(ratings[i]<ratings[i-1]) newSlope = -1;
            else newSlope = 0;


            if((oldSlope>0 && newSlope==0) || (oldSlope<0 && newSlope>=0)){
                candies+= sum(up) + sum(down) + Math.max(up,down);
                up=0;
                down=0;
            }

            if(newSlope==1) up++;
            else if(newSlope==-1) down++;
            else candies++;

            oldSlope = newSlope;

        }
        candies+= sum(up) + sum(down) + Math.max(up,down);
        return candies;
    }

    public int candy3(int[] ratings) {
        int candies = 1;
        int i=1;
        while(i<ratings.length){
            if(ratings[i]==ratings[i-1]) {
                candies++;i++;
                continue;
            }

            int peak =1;
            while(ratings[i]> ratings[i-1]){
                candies+=++peak;
                if(++i==ratings.length) return candies;
            }

            int valley = 1;
            while(i<ratings.length && ratings[i]<ratings[i-1]){
                candies+=++valley;
                i++;
            }
            if(peak>1 && valley>1){
                candies-=Math.min(peak, valley);
            }
        }
        return candies;

    }
    public boolean canFormString(String s, Map<String, Integer> hashWord, int wordLength){
        Map<String, Integer> hashCharactersInS = new HashMap<>();
        for(int i=0;i<s.length();i+=wordLength){
            String subStr = s.substring(i, i+wordLength);
            hashCharactersInS.put(subStr, hashCharactersInS.getOrDefault(subStr,0)+1);
        }
        return hashCharactersInS.equals(hashWord);
    }

    public List<Integer> findSubstring(String s, String[] words) {
        Map<Character,Integer> hashCharactersInWords = new HashMap<>();
        Map<String, Integer> hashWords = new HashMap<>();
        int wordsLength = 0;
        Set<Character> setWords = new HashSet<>();
        for(String word:words){
            hashWords.put(word, hashWords.getOrDefault(word,0)+1);
            for(int i=0;i<word.length();i++){
                hashCharactersInWords.put(word.charAt(i), hashCharactersInWords.getOrDefault(word.charAt(i),0)+1);
                wordsLength++;
                setWords.add(word.charAt(i));
            }
        }
        int uniqueWords = setWords.size();
        int formed = 0;
        Map<Character,Integer> hashCharactersInS = new HashMap<>();
        List<Integer> ans = new ArrayList<>();
        for(int i=0;i<s.length();i++){
            hashCharactersInS.put(s.charAt(i), hashCharactersInS.getOrDefault(s.charAt(i),0)+1);
            if(Objects.equals(hashCharactersInS.get(s.charAt(i)), hashCharactersInWords.get(s.charAt(i)))) formed++;
            if(uniqueWords==formed && canFormString(s.substring(i-wordsLength+1,i+1), hashWords,words[0].length())){
                ans.add(i-wordsLength+1);
            }
            if(i>=wordsLength){
                if(Objects.equals(hashCharactersInS.get(s.charAt(i-wordsLength)),
                        hashCharactersInWords.get(s.charAt(i-wordsLength)))) formed--;
                hashCharactersInS.put(s.charAt(i-wordsLength), hashCharactersInS.get(s.charAt(i-wordsLength))-1);
            }
        }
        return ans;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>(Comparator.comparingInt(l -> l.val));
        for(int i=0;i<lists.length;i++){
            pq.offer(lists[0]);
        }
        ListNode dummy = new ListNode(-1);
        ListNode preHead = dummy;
        while(!pq.isEmpty()){
            ListNode l = pq.poll();
            dummy.next = l;
            dummy = dummy.next;
            if(l.next!=null){
                pq.offer(l.next);
            }
        }
        return preHead.next;
    }








}
