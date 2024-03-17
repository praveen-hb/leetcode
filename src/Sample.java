import java.math.BigInteger;
import java.util.*;

public class Sample {
    public static void main (String args[]) {

        System.out.println(lengthOfLongestSubstring("abcabcdanc"));
    }
    
    
    public static int lengthOfLongestSubstring(String s) {
        int ans =0;
        int start =0;
        int i=0;
        Map<Character,Integer> uniqueString = new HashMap<>();
        for(i=0;i<s.length();i++){
            if(uniqueString.getOrDefault(s.charAt(i),-1) >= start){
                System.out.println(s.substring(start,i));
                ans = Math.max(ans,i-start);
                
                start = uniqueString.get(s.charAt(i)) +1;
                 System.out.println("" + ans + "," + start);
            }
            uniqueString.put(s.charAt(i),i);
        }
        ans = Math.max(ans, i-start);
        return ans;
    }
    
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        List<Integer> ans = new ArrayList<>();
        int i=0,j=0;
        while(i!=nums1.length || j!=nums2.length){
            if(i!=nums1.length && j!=nums2.length){
                if(nums1[i] < nums2[j]){
                    ans.add(nums1[i++]);
                } else{
                    ans.add(nums2[j++]);
                }
            } else if(nums2.length !=i){
                ans.add(nums2[j++]);
            } else{
                ans.add(nums1[i++]);
            }
        }
        
        int len = ans.size();
        if(len%2==0){
            return (double) (ans.get(len / 2 - 1) + ans.get(len / 2)) /2;
        } else 
            return ans.get(len/2);
    }
    
    public String sortVowels(String s) {
        List<Integer> consonents = new ArrayList<>();
        StringBuilder cons= new StringBuilder();
        for(int i=0;i<s.length();i++){
            if("aeiouAEIOU".indexOf(s.charAt(i)) > -1){
                consonents.add(i);
                cons.append(s.charAt(i));
            }
        }
        char[] charArray = cons.toString().toCharArray();
        Arrays.sort(charArray);
        StringBuilder ans = new StringBuilder(s);
        for(int i=0;i<charArray.length;i++){
            ans.setCharAt(consonents.get(i),charArray[i]);
        }
        return ans.toString();
    }
    
    public int countPalindromicSubsequence(String s) {
        int first[] = new int[26];
        int last[] = new int[26];
        Arrays.fill(first, -1);
        for(int i=0;i<s.length();i++){
            if(first[s.charAt(i)-'a']==-1){
                first[s.charAt(i)-'a']=i;
            }
            last[s.charAt(i)-'a']=i;
        }
        int ans=0;
        for(int i=0;i<26;i++){
            if(first[i]!=-1){
                Set<Character> between = new HashSet<>();
                for(int j=first[i]+1;j<last[i];j++){
                    between.add(s.charAt(j));
                }
                ans+=between.size();
            }
        }
        return ans;
        
    }
    
    public String convert(String s, int numRows) {
        if(numRows==1 || numRows>=s.length()) return s;
        StringBuilder result[] = new StringBuilder[numRows];
        for(int i=0;i<result.length;i++){
            result[i]= new StringBuilder();
        }
        int nextRow =1;
        int curRow=0;
        for(int i=0;i<s.length();i++){
            result[curRow].append(s.charAt(i));
            if(curRow==0) nextRow =1;
            if(curRow==numRows-1) nextRow =-1;
            curRow+= nextRow;
        }
        StringBuilder ans = new StringBuilder();
        for(StringBuilder sb : result){
            ans.append(sb);
        }
        return ans.toString();
    }
    
    public int maximumElementAfterDecrementingAndRearranging(int[] arr) {
        Arrays.sort(arr);
        arr[0]=1;
        int max = 1;
        for(int i=1;i<arr.length;i++){
            if(Math.abs(arr[i]-arr[i-1])>1){
                arr[i]=arr[i-1]+1;
            }
            max = Math.max(max, arr[i]);
        }
        return max;
    }
    public String generateBinary(String ans, Set<String> setNums, int n){
       
        if(ans.length() ==n ){
            if(!setNums.contains(ans)){
                return ans;
            }
            return "";
        }
        String comb = generateBinary(ans+"0",setNums, n);
        if(!comb.isEmpty()) return comb;
        return generateBinary(ans+"1",setNums,n);
          
    }
    
    public String findDifferentBinaryString(String[] nums) {
        Set<String> setNums = new HashSet<>(Arrays.asList(nums));
        String ans="";
        return generateBinary(ans, setNums, nums.length);
        
    }
    
    public int minPairSum(int[] nums) {
        Arrays.sort(nums);
        int max = Integer.MIN_VALUE;
        for(int i=0;i<nums.length;i++){
            max = Math.max(nums[i]+nums[nums.length-i-1],max);
        }
        return max;
    }
    
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, ArrayList<String>> groups = new HashMap<>();
        for(String str : strs){
            char[] charArray = str.toCharArray();
            Arrays.sort(charArray);
            String sortedStr = new String(charArray);
            groups.computeIfAbsent(sortedStr, k-> new ArrayList<>());
            groups.get(sortedStr).add(str);
        }
        
        return new ArrayList<>(groups.values());
    }
    
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int temp[]= Arrays.copyOf(nums1,m);
        int i=0,j=0;
        while(i<m || j<n){
            if(i==m) nums1[i+j] = nums2[j++];
            else if(j==n) nums1[i+j]= temp[i++];
            else if(temp[i]<nums2[j]) nums1[i+j]=temp[i++];
            else nums1[i+j]=nums2[j++];
        }
    }
    
    public int maxProfit(int[] prices) {
        int min = prices[0];
        int maxProfit = 0;
        for(int i=1;i<prices.length;i++){
            if(prices[i]<min){
                min=prices[i];
            } else {
                maxProfit=Math.max(maxProfit, prices[i]-min);
            }
        }
        return maxProfit;
    }
    
    public int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE;
        int csum =0;
        for(int i=0;i<nums.length;i++){
            maxSum = Math.max(maxSum,nums[i]);
            if(csum+nums[i]>0){
                csum+=nums[i];
                maxSum=Math.max(maxSum,csum);
            } else csum=0;
        }
        return maxSum;
        
    }
    
    public int maxProfit2(int[] prices) {
        int cprofit =0;
        for(int i=1;i<prices.length;i++){
            if(prices[i]-prices[i-1]>0){
                cprofit+=prices[i]-prices[i-1];
            }
        }
        return cprofit;
    }
    
    public int maxProfit3(int[] prices) {
        List<Integer> profits = new ArrayList<>();
        int cMin=prices[0],cMax=prices[0];
        for(int i=0;i<prices.length;i++){
            
            if(prices[i]<cMin){
                if(cMax>cMin){
                    profits.add(cMax-cMin);
                }
                cMin=prices[i];
                cMax=prices[i];
            }
            if(prices[i]>cMax){
                cMax = prices[i];
            }
        }
        if(cMax>cMin){
            profits.add(cMax-cMin);
        }
       
            int max1=0,max2=0;
            for(int i=0;i<profits.size();i++){
                if(max1 < profits.get(i)){
                    max2=max1;
                    max1=profits.get(i);
                } else if (max2<profits.get(i)) {
                    max2=profits.get(i);
                }
            }
            return max1+max2;
        
    }
    
    public int maxProfit4(int[] prices) {
        int buy1=Integer.MAX_VALUE;
        int buy2=Integer.MAX_VALUE;
        int sell1=0;
        int sell2=0;
        for(int price: prices){
            buy1=Math.min(buy1,price);
            sell1=Math.max(sell1,price-buy1);
            buy2=Math.min(buy2,price-sell1);
            sell2=Math.min(sell2,price-buy2);
        }
        return sell2;
    }
    
    public int maxProfit5(int k, int[] prices) {
        int buy[]=new int[k];
        int profit[]= new int[k];
        Arrays.fill(buy,Integer.MAX_VALUE);
        for(int price: prices){
            for(int i=0;i<k;i++){
                if(i==0){
                    buy[i]=Math.min(buy[i],price);
                } else {
                    buy[i]=Math.min(buy[i], price-buy[i-1]);
                }
                profit[i]=Math.max(profit[i],price-buy[i]);
            }
        }
        return profit[k-1];
    }
    
    public int maxFrequency(int[] nums, int k) {
        Arrays.sort(nums);
        int maxFrequency = 1;
        for(int i=0;i<nums.length;i++){
            int j=i-1, chances=k;
            int count =1;
            while(j>0 && chances>0){
                if(chances>=nums[i]-nums[j]){
                    chances-=nums[i]-nums[j];
                    count++;
                    j--;
                } else {
                    break;
                }
            }
            while(i+1<nums.length && nums[i+1]==nums[i]){
                count++;i++;
            }
            maxFrequency=Math.max(maxFrequency,count);
        }
        return maxFrequency;
    }
    
    TreeNode ans;
    
    public boolean findLCA(TreeNode root,TreeNode p,TreeNode q){
        if(root==null) return false;
        if(root==p || root==q) return true;
        boolean left = findLCA(root.left,p,q);
        boolean right = findLCA(root.right,p,q);
        boolean curr = (root==p || root==q);
        if((left && right) || (right && curr) || (curr && left)){
            ans=root;
        }
        return true;
    }
    
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==p || root==q) return root;
       findLCA(root,p,q);
       return ans;
    }
    public long summ(int n){
        return BigInteger.valueOf(n).multiply(BigInteger.valueOf(n+1)).divide(BigInteger.valueOf(2)).longValue();
    }
    
    public boolean isPossible(int n, int index, int maxSum, int high){
        long leftLeastSum=0,rightLeastSum=0;
        //Left sum calculation
        if(high > index){
            leftLeastSum = summ(high-1)-summ(high-index-1);
        } else {
            leftLeastSum = summ(high-1)+index-high+1;
        }
        
        if(high>n-index-1){
            rightLeastSum =summ(high-1)-summ(high-(n-index-1)-1);
        } else {
            rightLeastSum = summ(high-1)+n-index-high;
        }
        return leftLeastSum+rightLeastSum+high<=maxSum;
    }
    
    public int maxValue(int n, int index, int maxSum) {
       int left=1,right=maxSum;
       int cMax=1;
       while(left<right){
           int mid = (left+right+1)/2;
           if(isPossible(n,index,maxSum,mid)){
               cMax=mid;
               left=mid+1;
           } else {
               right=mid-1;
           }
       }
       return cMax;
    }
    
    public int reductionOperations(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int num: nums){
            set.add(num);
        }
        Arrays.sort(nums);
        set.remove(nums[0]);
        int ans=0;
        int cMax=nums[nums.length-1];
        for(int i=nums.length-1;i>0;i--){
            if(nums[i]!=cMax){
                set.remove(cMax);
                cMax=nums[i];
            }
            ans+=set.size();
        }
        return ans;
    }
    
    public long minimumReplacement(int[] nums) {
        long count=0;
        for(int i=nums.length-2;i>=0;i--){
            if(nums[i]>nums[i+1]){
                int numDivisions = nums[i]/nums[i+1];
                if(nums[i]%nums[i+1]!=0){
                    numDivisions++;
                }
                count+=numDivisions-1;
                nums[i]=nums[i]/numDivisions;
            }
        }
        return count;
    }
    
    public int minOperations(int[] nums) {
        int count=0;
        for(int i=1;i<nums.length;i++){
            if(nums[i]<nums[i-1]){
                count+=nums[i-1]-nums[i]+1;
                nums[i]=nums[i-1]+1;
            }
        }
        return count;
    }
    
    public int lengthOfLongestSubstring2(String s) {
        if(s.isEmpty()) return 0;
        int ans=1;
        int hash[]=new int[256];
        Arrays.fill(hash,-1);
        int start=0;
        hash[s.charAt(0)]=0;
        for(int i=1;i<s.length();i++){
            if(hash[s.charAt(i)]==-1){
                hash[s.charAt(i)]=i;
            } else{
                ans=Math.max(ans,i-start);
                start=i;
                hash[s.charAt(i)]=i;
            }
        }
        ans=Math.max(ans,s.length()-start);
        return ans;
    }
    
    public int garbageCollection(String[] garbage, int[] travel) {
        int countM=0,countP=0,countG=0;
        for(int i=0;i<garbage.length;i++){
            for(char ch: garbage[i].toCharArray()){
                if(ch=='M') countM++;
                if(ch=='P') countP++;
                if(ch=='G') countG++;
            }
        }
        int ans=0,currentGarbage=0;
        while(countM>0){
            if(currentGarbage!=0) ans+=travel[currentGarbage-1];
            for(char ch:garbage[currentGarbage].toCharArray()){
                if(ch=='M') {
                    ans++;
                    countM--;
                }
            }
            currentGarbage++;
        }
        currentGarbage=0;
        while(countP>0){
            if(currentGarbage!=0) ans+=travel[currentGarbage-1];
            for(char ch:garbage[currentGarbage].toCharArray()){
                if(ch=='P') {
                    ans++;
                    countP--;
                }
            }
            currentGarbage++;
        }
        currentGarbage=0;
        while(countG>0){
            if(currentGarbage!=0) ans+=travel[currentGarbage-1];
            for(char ch:garbage[currentGarbage].toCharArray()){
                if(ch=='G') {
                    ans++;
                    countG--;
                }
            }
            currentGarbage++;
        }
        return ans;
    }
    
    public int rev(int num){
        int ans=0;
        while(num>0){
            ans*=10;
            ans+=num%10;
            num/=10;
        }
        return ans;
    }
    
    public int countNicePairs(int[] nums) {
        for(int i=0;i<nums.length;i++){
            nums[i]= nums[i]-rev(nums[i]);
        }
        HashMap<Integer,Integer> freq= new HashMap<>();
        long ans=0;
        for(int i=0;i<nums.length;i++){
            ans+=freq.getOrDefault(nums[i],0);
            freq.put(nums[i], freq.getOrDefault(nums[i],0)+1);
            ans = ans%(1000000007);
        }
        return (int)ans;
    }
    
    public long countBadPairs(int[] nums) {
        int temp[]= nums.clone();
        int n=temp.length;
        for(int i=0;i<n;i++){
            temp[i]-=i;
        }
        long badPairsCount=0;
        Map<Integer,Long> freq= new HashMap<>();
        for(int i=0;i<n;i++){
            badPairsCount+=i-freq.getOrDefault(temp[i],0l);
            freq.put(temp[i],freq.getOrDefault(temp[i],0l)+1);
        }
        return badPairsCount;
    }
    
    public int[] findDiagonalOrder(List<List<Integer>> nums) {
       Map<Integer, List<Integer>> groups=new HashMap<>();
       int n=0;
       for(int i=nums.size()-1;i>=0;i--){
           for(int j=0;j<nums.get(i).size();j++){
               groups.computeIfAbsent(i+j, k-> new ArrayList<>());
               groups.get(i+j).add(nums.get(i).get(j));
               n++;
           }
       }
       
       int ans[]= new int[n];
       int curr =0;
       int i=0;
       while(groups.containsKey(curr)){
           for(int val:groups.get(curr)){
               ans[i++]=val;
           }
           curr++;
       }
       return ans;
    }
    
    public List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
        List<Boolean> ans = new ArrayList<>();
        for(int i=0;i<l.length;i++){
            int temp[]=Arrays.copyOfRange(nums, l[i],r[i]);
            Arrays.sort(temp);
            Boolean arithmetic = Boolean.TRUE;
            for(int j=2;j<temp.length;j++){
                if(temp[j]-temp[j-1]!=temp[j-1]-temp[j-2]){
                    arithmetic=Boolean.FALSE;
                    break;
                }
            }
            ans.add(arithmetic);
        }
        return ans;
    }
    
    public int maxCoins(int[] piles) {
        Arrays.sort(piles);
        int n=piles.length/3;
        int ans=0;
        for(int i=piles.length-2;n>0;i-=2,n--){
            ans+=piles[i];
        }
        return ans;
    }
    
    public int mostFrequent(int[] nums, int key) {
        int freq[]=new int[1001];
        int ansCount=0,ans=-1;
        for(int i=0;i<nums.length;i++){
            if(i!=nums.length-1 && nums[i]==key){
                freq[nums[i+1]]++;
                if(freq[nums[i+1]]>ansCount){
                    ansCount=freq[nums[i+1]];
                    ans=nums[i+1];
                }
            }
        }
        return ans;
    }
    
    public int[] getSumAbsoluteDifferences(int[] nums) {
        int prefix[] = nums.clone();
        for(int i=1;i<nums.length;i++){
            prefix[i]+=prefix[i-1];
        }
        int n=nums.length;
        int ans[] = new int[n];
        for(int i=0;i<n;i++){
            int leftSum =0,rightSum=0;
            if(i!=0){
                leftSum = i*nums[i] - prefix[i-1];
            }
            if(i!=n-1){
                rightSum = (n-i-1)*nums[i] - (prefix[n-1]-prefix[i]);
            }
            ans[i]= leftSum+rightSum;
        }
        return ans;
        
    }
    
    public int largestSubmatrix(int[][] matrix) {
        int ans = 0;
        int rows = matrix.length, columns = matrix[0].length;
        int prevRow[] = new int[columns];
        for(int i=0;i<rows;i++){
            int currRow[] = new int[columns];
            for(int j=0;j<columns;j++){
                if(matrix[i][j]==1){
                    currRow[i]= matrix[i][j]+prevRow[j];
                } else{
                    currRow[i]=0;
                }
            }
            prevRow = currRow.clone();
            Arrays.sort(currRow);
            for(int j=0;j<columns;j++){
                ans=Math.max(ans, currRow[j]*(columns-j));
            }
        }
        return ans;
    }
    
    public int hammingWeight(int n) {
        int numberOfOnes = 0;
        while(n>0){
          n = n & (n-1);
          numberOfOnes++;
        }
        return numberOfOnes;
    }
    
    public int[] sortByBits(int[] arr) {
        Integer arrInteger[] = Arrays.stream(arr).boxed().toArray(Integer[]::new);
        Arrays.sort(arrInteger, new CustomComparator());
        return Arrays.stream(arrInteger).mapToInt(x-> x).toArray();
    }
    
}

class CustomComparator implements Comparator<Integer> {
    
    @Override
    public int compare(Integer o1, Integer o2) {
        if(Integer.bitCount(o1) == Integer.bitCount(o2)){
            return o1-o2;
        }
        return Integer.bitCount(o1) - Integer.bitCount(o2);
    }
    
    public int knightDialer(int n) {
        int jumps[][] = new int[][]{
                {4,6},{6,8},{7,9},{4,8},{0,3,9},{},{0,1,7},{2,6},{1,3},{2,4}
        };
        int MOD = (int)1e9+7;
        int dp[][] = new int[n][10];
        Arrays.fill(dp[0],1);
        for(int i=1;i<n-1;i++){
            for(int place=0;place<10;place++){
                int ans=0;
                for(int nextJump:jumps[place]){
                    ans = (ans+ dp[i-1][nextJump])%MOD;
                }
                dp[i][place]=ans;
            }
        }
        int ans =0;
        for(int place=0;place<10;place++){
            ans = (ans+dp[n-1][place])%MOD;
        }
        return ans;
    }
    
    public void populateFreq(TreeNode root, Map<Integer,Integer> freq){
        freq.put(root.val,freq.getOrDefault(root.val,0)+1);
        if(root.left != null){
            populateFreq(root.left,freq);
        }
        if(root.right!=null){
            populateFreq(root.right,freq);
        }
        
    }
    
    public int[] findMode(TreeNode root) {
        Map<Integer,Integer> freq = new HashMap<>();
        populateFreq(root,freq);
        int maxFreq = Collections.max(freq.values());
        List<Integer> ansList = new ArrayList<>();
        for(int key: freq.keySet()){
            if(freq.get(key) == maxFreq){
                ansList.add(key);
            }
        }
        return ansList.stream().mapToInt(x->x).toArray();
    }
    
    public int[] processTree(TreeNode root){
        if(root==null) return new int[]{0,0,0};
        int leftRes[]= processTree(root.left);
        int rightRes[]= processTree(root.right);
        int sum = leftRes[0]+rightRes[0]+root.val;
        int count = leftRes[1]+rightRes[1]+1;
        int ans = leftRes[2]+rightRes[2];
        if(sum/count==root.val)ans++;
        return new int[]{sum,count,ans};
    }
    
    public int averageOfSubtree(TreeNode root) {
       int result[] = processTree(root);
       return result[2];
    }
    
    public List<String> buildArray(int[] target, int n) {
        List<String> ans = new ArrayList<>();
        int i=1;
        for(int element: target){
            while(i!=element){
                ans.add("Push");
                ans.add("Pop");
                i++;
            }
            ans.add("Push");
            i++;
        }
        return ans;
    }
    
    public int minOperations(List<Integer> nums, int k) {
        boolean freq[]=new boolean[k+1];
        int count=0;
        for(int i=nums.size()-1;i>=0;i--) {
            if(nums.get(i)<=k && !freq[nums.get(i)]){
                freq[nums.get(i)]=true;
                count++;
            }
            if(count==k) return nums.size()-i;
        }
        return -1;
    }
    
    public int getLastMoment(int n, int[] left, int[] right) {
        int maxLeft = Arrays.stream(left).max().orElse(0);
        int minRight = Arrays.stream(right).min().orElse(n);
        return Math.max(n-minRight,maxLeft);
    }
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        StringBuilder s1= new StringBuilder();
        StringBuilder s2= new StringBuilder();
        for(String s : word1){
            s1.append(s);
        }
        for(String s:word2){
            s2.append(s);
        }
        return s1.toString().contentEquals(s2);
                
    }
    
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        List<List<Integer>> ans = new ArrayList<>();
        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
        Set<Map.Entry<Integer,Integer>> visited = new HashSet<>();
        queue.offer(new int[]{nums1[0]+nums2[0],0,0});
        visited.add(new AbstractMap.SimpleEntry<>(0,0));
        while(k>0 && !queue.isEmpty()){
            int pair[] = queue.poll();
            int i=pair[1],j= pair[2];
            List<Integer> oneAns = new ArrayList<>();
            oneAns.add(nums1[i]);
            oneAns.add(nums2[j]);
            ans.add(oneAns);
            if(i+1<nums1.length && !visited.contains(new AbstractMap.SimpleEntry<>(i+1,j))){
                queue.offer(new int[]{nums1[i+1]+nums2[j],i+1,j});
                visited.add(new AbstractMap.SimpleEntry<>(i+1,j));
            }
            if(j+1<nums2.length && !visited.contains(new AbstractMap.SimpleEntry<>(i,j+1))){
                queue.offer(new int[]{nums1[i]+nums2[j+1],i,j+1});
                visited.add(new AbstractMap.SimpleEntry<>(i,j+1));
            }
            k--;
        }
        return ans;
 
    }
    
    public int countCharacters(String[] words, String chars) {
        int freq[]= new int[26];
        for(char c:chars.toCharArray()){
            freq[c-'a']++;
        }
        int ansCount=0;
        for(String word: words){
            boolean goodString=true;
            int freqCopy[] = freq.clone();
            for(char c:word.toCharArray()){
                if(freqCopy[c-'a']==0){
                    goodString=false;
                    break;
                } else {
                    freqCopy[c-'a']--;
                }
            }
            if(goodString) ansCount+=word.length();
        }
        return ansCount;
    }
    
    public List<List<Integer>> kSmallestPairs2(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        
        int left = nums1[0] + nums2[0];
        int right = nums1[len1 - 1] + nums2[len2 - 1];
        while (left <= right) {
            int middle = left + (right - left) / 2;
            
            long cnt = getSmallerGreaterCnt(nums1, nums2, middle, k);
            if (cnt < k) {
                left = middle + 1;
            } else if (cnt > k) {
                right = middle - 1;
            } else {
                left = middle;
                break;
            }
        }
        return getPairs(nums1, nums2, left, k);
    }
    
    int getSmallerGreaterCnt(int[] nums1, int[] nums2, int target, int k) {
        int previousRight = nums2.length - 1;
        int cnt = 0;
        for (int i = 0; i < nums1.length && nums1[i] + nums2[0] <= target; i++) {
            int left = 0;
            int right = previousRight;
            int pos = -1;
            while (left <= right) {
                int middle = (left + right) / 2;
                int sum = nums1[i] + nums2[middle];
                if (sum <= target) {
                    pos = middle;
                    left = middle + 1;
                } else {
                    right = middle - 1;
                }
            }
            if (pos >= 0) {
                cnt += pos + 1;
                previousRight = pos;
            }
            if (cnt > k) {
                return cnt;
            }
        }
        return cnt;
    }
    
    List<List<Integer>> getPairs(int[] nums1, int[] nums2, int targetSum, int k) {
        List<List<Integer>> pairs = new ArrayList<>();
        for (int first : nums1) {
            for (int j = 0; j < nums2.length && first + nums2[j] < targetSum; j++) {
                pairs.add(Arrays.asList(first, nums2[j]));
            }
        }
        for (int first : nums1) {
            for (int j = 0; j < nums2.length && first + nums2[j] <= targetSum
                    && pairs.size() < k; j++) {
                if (first + nums2[j] == targetSum) {
                    pairs.add(Arrays.asList(first, nums2[j]));
                }
            }
        }
        return pairs;
    }

    
    public int minTimeToVisitAllPoints(int[][] points) {
        int time =0;
        for(int i=1;i<points.length;i++){
            time+=Math.max(Math.abs(points[i][0]-points[i-1][0]),Math.abs(points[i][1]-points[i-1][1]));
        }
        return time;
    }
    
    public double myPow(double x, int n) {
        if(n==0) return 1;
        if(n<0) {
            x=1.0/x;
            n=-n;
        }
        double res=1.0;
        while(n!=0){
            if(n%2==0){
                x=x*x;
                n/=2;
            } else {
                res=res*x;
                n=n-1;
            }
        }
        return res;
    }
    
    public int longestNiceSubarray(int[] nums) {
        int ans=1;
        int cur=nums[0];
        int start =0;
        for(int i=1;i<nums.length;i++){
            while(start!=i && (cur&nums[i])!=0){
                cur^=nums[start++];
            }
            cur|=nums[i];
            ans=Math.max(ans,i-start+1);
        }
        return ans;
    }
    
    public boolean isPossible(String string, Set<String> wordSet, Map<String,Boolean> hash){
        if(hash.containsKey(string)){
            return hash.get(string);
        }
       if(string.length()==1){
           hash.put(string,wordSet.contains(string));
           return wordSet.contains(string);
       }
        for(int i=1;i<string.length()+1;i++){
            String firstWord = string.substring(0,i);
            if(wordSet.contains(firstWord)){
                if(i==string.length() || isPossible(string.substring(i),wordSet,hash)){
                    hash.put(string,Boolean.TRUE);
                    return true;
                }
            }
        }
        hash.put(string,Boolean.FALSE);
        return false;
    }
    
    public boolean wordBreak(String s, List<String> wordDict) {
        Map<String,Boolean> hash = new HashMap<>();
        Set<String> wordSet = new HashSet<>(wordDict);
        return isPossible(s,wordSet,hash);
    }
    
    public int totalMoney(int n) {
        int numberOfCompletedWeeks = n/7;
        int numberOfDaysOutsideWeek = n%7;
        return numberOfCompletedWeeks*28 + 7*(numberOfCompletedWeeks*(numberOfCompletedWeeks-1))/2 + numberOfCompletedWeeks*numberOfDaysOutsideWeek + ((numberOfDaysOutsideWeek)*(numberOfDaysOutsideWeek+1))/2;
    }
    
    private void generateAns(TreeNode root, StringBuilder ans){
        if(root==null) return;
        ans.append(root.val);
        if(root.left != null){
            ans.append("(");
            generateAns(root.left,ans);
            ans.append(")");
        }
        if(root.right!=null){
            ans.append("(");
            generateAns(root.right, ans);
            ans.append(")");
        }
    }
    
    public String tree2str(TreeNode root) {
        StringBuilder ans = new StringBuilder("");
        generateAns(root, ans);
        return ans.toString();
    }
    
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(int i=0;i<s.length();i++){
            Character c = s.charAt(i);
            if(c=='(' || c=='{' || c=='['){
                stack.push(c);
            } 
            if(stack.isEmpty()) return false;
            Character top = stack.pop();
            if(c=='}' && top!='{') return false;
            if(c==')' && top!='(') return false;
            if(c==']' && top!='[') return false;
        }
        return stack.isEmpty();
    }
    

    public int[][] imageSmoother(int[][] img) {
        int r=img.length;
        int c=img[0].length;
        int ans[][] = new int[r][c];
        int[][] twoDArray = {
                {-1,-1}, {-1, 0}, {-1, 1}, {0,-1}, {0,1},{1,-1},{1,0},{1,1}
        };
        List<List<Integer>> listOfLists = new ArrayList<>(Arrays.asList(
                Arrays.asList(1, 2, 3),
                Arrays.asList(4, 5, 6)
        ));
        return ans;
    }
    
    public int maxWidthOfVerticalArea(int[][] points) {
        Arrays.sort(points, Comparator.comparingInt(o -> o[0]));
        int ans = 0;
        for(int i=1;i<points.length;i++){
            ans=Math.max(ans,points[i][0]-points[i-1][0]);
        }
        return ans;
        
    }
    
    Map<String,Integer> foodRating = new HashMap<>();
    Map<String, List<String>> hash = new HashMap<>();
    public void foodRatings(String[] foods, String[] cuisines, int[] ratings) {
        for(int i=0;i<foods.length;i++){
            foodRating.put(foods[i],ratings[i]);
            hash.computeIfAbsent(cuisines[i], k -> new ArrayList<>()).add(foods[i]);
        }
    }
    
    public void changeRating(String food, int newRating) {
        foodRating.put(food,newRating);
    }
    
    public String highestRated(String cuisine) {
        List<String> foods = hash.get(cuisine);
        Collections.sort(foods, (o1, o2) -> {
            if(Objects.equals(foodRating.get(o1), foodRating.get(o2))){
                return o1.compareTo(o2);
            }
            else 
                return foodRating.get(o1) - foodRating.get(o2);
        });
        return foods.get(0);
    }

    public int findMin(int[] jobDifficulty, int curIndex, int remD, int memo[][]){
        if(remD==0 ||
                jobDifficulty.length == curIndex ||
                (remD > (jobDifficulty.length-curIndex))) {
            return -1;
        }
        int ans = Integer.MAX_VALUE;
        int cMax = Integer.MIN_VALUE;
        if(remD==1){
            for(int i=curIndex;i<jobDifficulty.length;i++){
                cMax = Math.max(cMax,jobDifficulty[i]);
            }
            return memo[remD][curIndex] = cMax;
        } else{
            for(int i=curIndex;i<jobDifficulty.length && (jobDifficulty.length - i)>=(remD-1);i++){
                cMax = Math.max(cMax,jobDifficulty[i]);
                int sol = findMin(jobDifficulty, i+1,remD-1,memo);
                if(sol!=-1){
                    ans = Math.min(ans,cMax+sol);
                }
            }
        }
        if(ans == Integer.MAX_VALUE) ans = -1;
        return memo[remD][curIndex]=ans;
    }

    public int minDifficulty(int[] jobDifficulty, int d) {
        int njobs = jobDifficulty.length;
        if(d> njobs) return -1;
        int memo[][] = new int[d+1][jobDifficulty.length];
        return findMin(jobDifficulty,0,d,memo);
    }

    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer,Integer> hash = new HashMap<>();
        for (int j : arr) {
            hash.put(j, hash.getOrDefault(j, 0) + 1);
        }
        Set<Integer> set = new HashSet<>();
        for(int key: hash.keySet()){
            if(set.contains(hash.get(key))){
                return false;
            } else{
                set.add(hash.get(key));
            }
        }
        return true;
    }


    public int calculateDp(int[][] matrix, int currRow, int currCol, int[][] dp){
        if(currRow >= matrix.length || currCol >= matrix[0].length || currCol<0) return 0;
        if(dp[currRow][currCol]!= Integer.MAX_VALUE) return dp[currRow][currCol];
        int dL = Integer.MAX_VALUE, dR = Integer.MAX_VALUE;
        if(currCol-1>=0){
            dL = calculateDp(matrix,currRow+1,currCol-1, dp);
        }
        int dB = calculateDp(matrix, currRow+1,currCol,dp);
        if(currCol+1<=matrix[0].length){
            dR = calculateDp(matrix,currRow+1,currCol+1,dp);
        }
        return dp[currRow][currCol] = Math.min(Math.min(dL,dB),dR)+matrix[currRow][currCol];
    }

    public int minFallingPathSum(int[][] matrix) {
        int r= matrix.length, c = matrix[0].length;
        int[][] dp = new int[r][c];
        for(int i=0;i<r;i++){
            Arrays.fill(dp[i],Integer.MAX_VALUE);
        }
        int ans= Integer.MAX_VALUE;
        for(int j=0;j<c;j++){
            ans = Math.min(ans, calculateDp(matrix,0,j,dp));

        }

        return ans;
    }

    public int findJudge(int n, int[][] trust) {
        int[] trustees = new int[n+1];
        boolean[] isTrustExists = new boolean[n+1];
        Arrays.fill(trustees,0);
        Arrays.fill(isTrustExists,false);
        for(int[] pair : trust){
            isTrustExists[pair[0]]= true;
            trustees[pair[1]]++;
        }
        for(int i=1;i<n+1;i++){
            if(trustees[i]==n-1 && !isTrustExists[i])
                return i;
        }
        return -1;

    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        Map<Integer, List<int[]>> adj = new HashMap<>();
        for(int[] flight : flights){
            adj.computeIfAbsent(flight[0], x -> new ArrayList<>()).add(new int[]{flight[1],flight[2]});
        }

        int[] stops = new int[n];
        Arrays.fill(stops, Integer.MAX_VALUE);
        stops[src] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        pq.add(new int[]{src,0,0});
        while(!pq.isEmpty()){
            int[] curNode = pq.remove();
            int sourceNode = curNode[0];
            int curCost = curNode[1];
            int steps = curNode[2];
            if(sourceNode == dst)
                return curCost;
            if(steps>stops[sourceNode] || steps>k)
                continue;
            for(int[] adjNode : adj.getOrDefault(curNode[0], Collections.emptyList())){
               pq.add(new int[]{adjNode[0], adjNode[1]+ curCost, steps+1});
               stops[adjNode[0]]= steps+1;
            }

        }
        return -1;

    }

    public List<Integer> findAllPeople(int n, int[][] meetings, int firstPerson) {
        Map<Integer, List<int[]> > meet = new HashMap<>();
        for(int[] meets : meetings){
            meet.computeIfAbsent(meets[0], x-> new ArrayList<>()).add(new int[]{meets[1], meets[2]});
            meet.computeIfAbsent(meets[1], x-> new ArrayList<>()).add(new int[]{meets[0], meets[2]});
        }

        boolean[] visited = new boolean[n];
        Arrays.fill(visited, false);
        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[1]));
        pq.offer(new int[]{0,0});
        pq.offer(new int[]{firstPerson, 0});
        while(!pq.isEmpty()){
            int[] personTime = pq.poll();
            int person = personTime[0];
            int currTime = personTime[1];
            if(visited[person]){
                continue;
            }
            visited[person]= true;
            for(int[] curMeet : meet.getOrDefault(person, new ArrayList<>())){
                if(curMeet[1]>=currTime){
                    pq.offer(new int[]{curMeet[0], curMeet[1]});
                }
            }
        }
        List<Integer> ans = new ArrayList<>();
        for(int i=0;i<visited.length;i++){
            if(visited[i]){
                ans.add(i);
            }
        }
        return ans;
    }

    public int gcd(int a, int b){
        if(a < b){
            int temp = a;
            a=b;
            b=temp;
        }
        if(a % b ==0) return b;
        else return gcd(b, a%b);
    }

    public boolean canTraverseAllPairs(int[] nums) {
        int n= nums.length;
        if(n==1) return true;
        Stack<Integer> stack = new Stack<>();
        Set<Integer> set = new HashSet<>();
        for(int nu : nums){
            set.add(nu);
        }
        stack.push(nums[0]);
        set.remove(nums[0]);
        while(!stack.isEmpty()){
            int num = stack.pop();
            Set<Integer> setCopy = new HashSet<>(set);
            for(int nu : set){
                if(gcd(num, nu) > 1){
                    stack.push(nu);
                    setCopy.remove(nu);
                }
            }
            if(setCopy.isEmpty()) return true;
            set = setCopy;
        }
        return false;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q ==  null) return false;
        if(p.val != q.val) return false;
        return isSameTree(p.left,q.left) && isSameTree(p.right,q.right);
    }


    public boolean isEvenOddTree(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int level =0;
        while(!queue.isEmpty()){
            int size = queue.size();
            int oddMax = Integer.MIN_VALUE;
            int evenMin = Integer.MAX_VALUE;
            for(int i=0;i<size;i++){
                TreeNode cur = queue.poll();
                if(level %2 ==0){
                    if(cur.val %2 ==0 || cur.val <= oddMax) return false;
                    oddMax = cur.val;
                } else{
                    if(cur.val % 2 !=0 || cur.val >= evenMin) return false;
                    evenMin = cur.val;
                }
                if(cur.left !=null) queue.add(cur.left);
                if(cur.right!= null) queue.add(cur.right);
            }
            level++;
        }
        return true;
    }

    public void dfs(TreeNode root, int[] ans, int level){
        if(level > ans[0]){
            ans[0] = level;
            ans[1] = root.val;
        }
        if(root.left != null) dfs(root.left, ans, level+1);
        if(root.right != null) dfs(root.right,ans, level+1);
    }

    public int findBottomLeftValue(TreeNode root) {
        int[] ans = new int[2];
        ans[0] = Integer.MIN_VALUE;
        ans[1] = -1;
        dfs(root, ans, 0);
        return ans[1];

    }

    public int[] sortedSquares(int[] nums) {
       int positiveIndex = -1;
       for(int i=0;i<nums.length;i++){
           if(nums[i]>=0){
               positiveIndex = i;
               break;
           }
       }
       int[] ans = new int[nums.length];
       int negativeIndex = positiveIndex - 1;
       while(negativeIndex > -1 && positiveIndex < nums.length){
               if(Math.abs(nums[negativeIndex]) >= nums[positiveIndex]){
                   ans[positiveIndex - negativeIndex -1] = nums[negativeIndex] * nums[negativeIndex];
                   negativeIndex--;
               } else{
                   ans[positiveIndex - negativeIndex -1] = nums[positiveIndex] * nums[positiveIndex];
               }
       }

       while(negativeIndex > -1){
           ans[positiveIndex - negativeIndex -1] = nums[negativeIndex] * nums[negativeIndex--];
       }
       while(positiveIndex < nums.length){
           ans[positiveIndex - negativeIndex -1] = nums[positiveIndex] * nums[positiveIndex++];
       }
        return ans;
    }

    public int[][] divideArray(int[] nums, int k) {
        int[][] ans = new int[nums.length/3][3];
        Arrays.sort(nums);
        for(int i=0;i<nums.length;i+=3){
            if(Math.abs(nums[i]-nums[i+1])<= k && Math.abs(nums[i]-nums[i+2])<=k){
                ans[i/3][0] = nums[i];
                ans[i/3][1]= nums[i+1];
                ans[i/3][2] = nums[i+2];
            } else {
                return new int[nums.length/3][3];
            }
        }
        return ans;
    }

    public int getDigits(int num){
        int count =0;
        while(num>0){
            count++;
            num/=10;
        }
        return count;
    }

    public int getLowestSequentialNumber(int digit){
        int ans =0;
        int counter =1;
        while(counter <= digit){
            ans= ans*10 + counter;
            counter++;
        }
        return ans;
    }

    public int getIncrement(int digit){
        int ans =0;
        while(digit-- > 0){
            ans = ans* 10 + 1;
        }
        return ans;
    }
    public List<Integer> sequentialDigits(int low, int high) {
        int lowD = getDigits(low);
        int highD = getDigits(high);
        List<Integer> ans = new ArrayList<>();
        for(int i=lowD;i<=highD;i++){
            int num = getLowestSequentialNumber(i);
            int increment = getIncrement(i);
            int numberOfPossibilities = 10 - i;
            do{
                if(num>= low && num<= high){
                    ans.add(num);
                }
                num+=increment;
            } while (--numberOfPossibilities>0);
        }
        return ans;
    }

    public int maxSumAfterPartitioning(int[] arr, int k) {
        int[] dp = new int[k];
        for(int i=arr.length-1;i>=0;i++){
            int end = Math.min(arr.length, i+k);
            int localMax = arr[i];
            for(int j = i; j<end;j++){
                localMax = Math.max(arr[j], localMax);
                dp[i%k] = Math.max(dp[i], localMax*(j-i+1) + dp[(j+1)%k]);
            }
        }
        return dp[0];
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode newhead = head;
        ListNode cur = head;
        while(n-- > 0){
            head = head.next;
        }
        if(head.next == null) return cur.next;

        while(head.next != null){
            head = head.next;
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return newhead;
    }

    public String minWindow(String s, String t) {
        if(s.isEmpty() || t.isEmpty()) return "";
        int[] freqT = new int[52];
        int uniqueT = 0;
        for(int i=0;i<t.length();i++){
            if(freqT[t.charAt(i)-'A']++==0) uniqueT++;
        }
        int formed =0;
        int[] freqS = new int[52];
        int[] ans = new int[]{-1,-1,-1};
        int left =0, right =0;
        while(right<s.length()){
            char c = s.charAt(right);
            if(++freqS[c-'A']==freqT[c-'A']) formed++;
            while(formed==uniqueT && left<=right){
                if(ans[0]==-1 || right-left+1 < ans[0]){
                    ans[0]= right-left+1;
                    ans[1]= left;
                    ans[2] = right;
                }
                char leftC = s.charAt(left);
                if(--freqS[leftC-'A']<freqT[leftC-'A']) formed--;
                left ++;
            }
            right++;
        }
        return ans[0]==-1 ? "" : s.substring(ans[1], ans[2]+1);
    }



    public int bagOfTokensScore(int[] tokens, int power) {
       Arrays.sort(tokens);
       int left =0, right = tokens.length-1;
       int curScore = 0;
       int maxScore = 0;
       while(left<=right){
           if(tokens[left]<power || (left==right && tokens[left]==power)){
               power-=tokens[left];
               left++;
               curScore++;
               maxScore = Math.max(maxScore, curScore);
           } else if(curScore!=0){
               power+=tokens[right];
               curScore--;
               right--;
           } else{
               break;
           }
       }
       return maxScore;
    }

    public <K,V extends Comparable<? super  V>> Map<K,V> sortByValues(Map<K, V> map){
        List<Map.Entry<K, V>> entries = new ArrayList<>(map.entrySet());
        entries.sort((o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        Map<K,V> sortedMap = new LinkedHashMap<>();
        for(Map.Entry<K,V> entry: entries){
            sortedMap.put(entry.getKey(), entry.getValue());
        }
        return sortedMap;

    }

    public String frequencySort(String s) {
        HashMap<Character,Integer> hash = new HashMap<>();
        for(char c: s.toCharArray()){
            hash.put(c, hash.getOrDefault(c,0)+1);
        }
        Map<Character,Integer> sortedMap = sortByValues(hash);
        StringBuilder ans = new StringBuilder();
        for(Map.Entry<Character,Integer> entry : sortedMap.entrySet()){
            for(int i=0;i<entry.getValue();i++){
                ans.append(entry.getKey());
            }
        }
        return new String(ans);

    }

    public String frequencySort1(String s) {
        HashMap<Character,Integer> hash = new HashMap<>();
        for(char c: s.toCharArray()){
            hash.put(c, hash.getOrDefault(c,0)+1);
        }
        List<Map.Entry<Character,Integer>> sortedList = new ArrayList<>(hash.entrySet());
        sortedList.sort((entry1,entry2) -> entry2.getValue().compareTo(entry1.getValue()));
        StringBuilder ans = new StringBuilder();
        for(Map.Entry<Character,Integer> entry : sortedList){
            for(int i=0;i<entry.getValue();i++){
                ans.append(entry.getKey());
            }
        }
        return new String(ans);

    }

    public int minimumLength(String s) {
        int left =0, right = s.length()-1;
        while(left<=right && s.charAt(left)==s.charAt(right)){
            char c = s.charAt(left);
            while(left<=right && c==s.charAt(left)){
                left++;
            }
            while(left<=right && c==s.charAt(right)){
                right--;
            }
        }
        return right-left+1;
    }

    public int numSquares(int n) {
        List<Integer> perfectSquares = new ArrayList<>();
        for(int i=1;i<=Math.sqrt(n);i++){
            perfectSquares.add(i*i);
        }
        int[] dp = new int[n+1];
        for(int i=1;i<=n;i++){
            int sol = Integer.MAX_VALUE;
            for(int square : perfectSquares){
                if(square > i) break;
                sol = Math.min(sol, dp[i-square] + 1);
            }
            dp[i]= sol;
        }
        return dp[n];
    }

    public List<Integer> largestDivisibleSubset(int[] nums) {
        Arrays.sort(nums);
        int[] dp = new int[nums.length];
        dp[0]=1;
        int ansMax = 1;
        int maxIndex = 0;
        for(int i=1;i<nums.length;i++){
            int cMax = 0;
            for(int j=0;j<i;j++){
                if(nums[j]%nums[i]==0) cMax = Math.max(cMax,dp[j]);
            }
            dp[i] = cMax+1;
            if(dp[i]>ansMax){
                ansMax= dp[i];
                maxIndex = i;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for(int i=maxIndex;i>=0;i--){
            if(nums[maxIndex]%nums[i]==0 && dp[i]==ansMax) {
                ans.add(nums[i]);
                ansMax--;
            }
        }
        return ans;
    }

    public int addPalinCount(String s, int left, int right){
        int ans =0;
        while(left>=0 && right<s.length()){
            if(s.charAt(left)!=s.charAt(right)) break;
            ans++;
            left--;
            right++;
        }
        return ans;
    }

    public int countSubstrings(String s) {
        int ans = 0;
        for(int i=0;i<s.length();i++){
            ans += addPalinCount(s, i, i);
            ans += addPalinCount(s, i, i+1);
        }
        return ans;
    }

    public void swap(int[] nums, int i, int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void sortColors(int[] nums) {
        int left =0, right = nums.length-1;
        for(int i=0;i<=right;i++){
            if(nums[i]==0) swap(nums, i, left++);
            if(nums[i]==2) swap(nums, i--, right--);
        }

    }

    public int numUniqueEmails(String[] emails) {
        Set<String> set = new HashSet<>();
        for(String email: emails){
            String[] split = email.split("@");
            String userName = split[0];
            String domain = split[1];
            userName = userName.replace(".","").split("\\+")[0];
            set.add(userName+domain);
        }
        return set.size();
    }

    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
       Stack<Integer> stack = new Stack<>();
       Map<Integer, Integer> hash = new HashMap<>();
       for(int n: nums2){
           while(!stack.isEmpty() && stack.peek()< n){
               hash.put(stack.pop(), n);
           }
           stack.push(n);
       }
       while(!stack.isEmpty()){
           hash.put(stack.pop(), -1);
       }
       int[] ans = new int[nums1.length];
       for(int i=0;i<nums1.length;i++){
           ans[i] = hash.get(nums1[i]);
       }
       return ans;
    }

    public int nextGreaterElement(int n) {
        char[] charArray = Integer.toString(n).toCharArray();
        Stack<Character> stack = new Stack<>();
        int replaceIndex = -1;
        List<Character> replaceChars = new ArrayList<>();
        for(int i=charArray.length-1;i>=0;i--){
            if(!stack.isEmpty() && stack.peek() > charArray[i]){

                char replaceWith = stack.peek() ;
                while(!stack.isEmpty() && stack.peek() > charArray[i]){
                    replaceWith = stack.pop();
                    replaceChars.add(replaceWith);
                }
                replaceChars.remove(replaceChars.size()-1);
                while(!stack.isEmpty()){
                    replaceChars.add(stack.pop());
                }
                replaceChars.add(charArray[i]);
                charArray[i] = replaceWith;
                replaceIndex = i;
                break;
            }
            stack.push(charArray[i]);
        }
        if(replaceIndex == -1 ) return -1;
        Collections.sort(replaceChars);
        for(int i=replaceIndex+1;i<charArray.length;i++){
            charArray[i] = replaceChars.get(i-replaceIndex-1);
        }
        long ansLong = Long.parseLong(new String(charArray));
        if(ansLong<= Integer.MAX_VALUE) return (int) ansLong;
        return -1;

    }

    public boolean canJump(int[] nums) {
        boolean[] dp = new boolean[nums.length];
        dp[nums.length-1] = true;
        for(int i= nums.length -2 ;i>=0;i--){
            for(int jump= 1; jump<= nums[i];jump++){
                if(dp[i+jump]==true) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[0];
    }




    public int oddEvenJumps(int[] arr) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        int n = arr.length;
        int[] oddJumps = new int[n];
        int[] evenJumps = new int[n];
        Arrays.fill(oddJumps, -1);
        Arrays.fill(evenJumps, -1);
        map.put(arr[n-1], n-1);
        for(int i=n-2;i>=0;i--){
            Map.Entry<Integer, Integer> ceilingEntry = map.ceilingEntry(arr[i]);
            if(ceilingEntry!=null){
                oddJumps[i] = ceilingEntry.getValue();
            }
            Map.Entry<Integer, Integer> floorEntry = map.floorEntry(arr[i]);
            if(floorEntry != null){
                evenJumps[i] = floorEntry.getValue();
            }
            map.put(arr[i], i);
        }
        System.out.print("Even Jumps : ");
        for(int i=0;i<n;i++){
            System.out.print(evenJumps[i]);
            System.out.print(" ");
        }
        System.out.println();

        System.out.print("Odd Jumps : ");
        for(int i=0;i<n;i++){
            System.out.print(oddJumps[i]);
            System.out.print(" ");
        }

        // 0 - Even , 1 - ODD
        boolean[][] dp = new boolean[n][2];
        dp[n-1][0] = true;
        dp[n-1][1] = true;
        int ans = 1;
        for(int i=n-2;i>=0;i--){
            if(evenJumps[i]!=-1){
                dp[i][0] = dp[evenJumps[i]][1];
            }
            if(oddJumps[i]!=-1){
                dp[i][1] = dp[oddJumps[i]][0];
            }
            if(dp[i][1]) ans++;
        }
        return ans;
    }

    public int[] nextGreaterElements(int[] nums){
        int[] arr = new int[nums.length];
        Stack<Integer> stack = new Stack<>();
        for(int i=0;i<nums.length;i++){
            while(!stack.isEmpty() && nums[stack.peek()] > nums[i]){
                arr[stack.pop()] = nums[i];
            }
            stack.push(i);
        }

        while(!stack.isEmpty()){
            arr[stack.pop()] = -1;
        }
        return arr;
    }


    public int oddEvenJumps2(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i=0;i<arr.length;i++){
            map.put(i, arr[i]);
        }
        List<Map.Entry<Integer, Integer>> valueToIndex = new ArrayList<>(map.entrySet());

        valueToIndex.sort(Comparator.comparingInt(Map.Entry::getValue));
        int[] ascArray = valueToIndex.stream().map(Map.Entry::getKey).mapToInt(Integer::intValue).toArray();
        int[] oddJumpMaps = nextGreaterElements(ascArray);
        int[] oddJumps = new int[arr.length];
        for(int i=0;i<arr.length;i++){
            oddJumps[ascArray[i]] = oddJumpMaps[i];
        }

        valueToIndex.sort((o1, o2) -> o2.getValue() - o1.getValue());
        int[] dscArray = valueToIndex.stream().map(Map.Entry::getKey).mapToInt(Integer::intValue).toArray();
        int[] evenJumpMaps = nextGreaterElements(dscArray);
        int[] evenJumps = new int[arr.length];
        for(int i=0;i<arr.length;i++){
            evenJumps[dscArray[i]] = evenJumpMaps[i];
        }

        // 0- ODD, 1 - Even
        boolean[][] dp = new boolean[arr.length][2];
        dp[arr.length-1][0] = true;
        dp[arr.length-1][1] = true;
        int ans = 1;
        for(int i=arr.length-2;i>=0;i--){
            if(evenJumps[i]!=-1){
                dp[i][1] = dp[evenJumps[i]][1];
            }
            if(oddJumps[i]!=-1){
                dp[i][0] = dp[oddJumps[i]][0];
            }
            if(dp[i][0]) ans++;
        }

        return ans;
    }


    public int oddEvenJumps3(int[] arr) {
        int[] count = new int[100001];
        boolean[][] res = new boolean[2][arr.length];
        res[0][arr.length - 1] = true;
        res[1][arr.length - 1] = true;
        count[arr[arr.length - 1]] = arr.length;
        int min = arr[arr.length - 1], max = arr[arr.length - 1];
        int result = 1;
        for(int i = arr.length - 2; i >= 0; i--) {
            int nextSmallIndex = findNextSmall(count, min, max, arr[i]);
            int nextLargeIndex = findNextLarge(count, min, max, arr[i]);
            if(nextSmallIndex == -1) {
                res[0][i] = false;
            } else {
                res[0][i] = res[1][nextSmallIndex];
            }
            if(nextLargeIndex == -1) {
                res[1][i] = false;
            } else {
                res[1][i] = res[0][nextLargeIndex];
            }
            count[arr[i]] = i + 1;
            min = Math.min(min, arr[i]);
            max = Math.max(max, arr[i]);
            if(res[0][i]) {
                result++;
            }
        }
        return result;
    }
    int findNextSmall(int[] count, int min, int max, int val) {

        for(int i=val; i <= max; i++) {
            if(count[i] != 0) {
                return count[i]-1;
            }
        }
        return -1;
    }
    int findNextLarge(int[] count, int min, int max, int val) {

        for(int i=val; i >= min; i--) {
            if(count[i] != 0) {
                return count[i]-1;
            }
        }
        return -1;
    }

    public int getOddNextIndex(int[] arr, int max, int val, int[] valToIndex){
        for(int i=val;i<=max;i++){
            if(valToIndex[i]!=0) return valToIndex[i];
        }
        return -1;
    }

    public int getEvenNextIndex(int[] arr, int min, int val, int[] valToIndex){
        for(int i=min;i<=val;i++){
            if(valToIndex[i]!=0) return valToIndex[i];
        }
        return -1;
    }

    public int oddEvenJumps4(int[] arr){
        int[] valToIndex = new int[100001];
        int n = arr.length;
        // 0 -> even, 1-> odd
        boolean[][] dp = new boolean[n][2];
        dp[n-1][0] = true;
        dp[n-1][1] = true;
        int ans = 1;
        valToIndex[arr[n-1]] = n-1;
        int min = arr[n-1], max = arr[n-1];
        for(int i=n-2;i>=0;i--){
            int oddNextIndex = getOddNextIndex(arr, max, arr[i], valToIndex);
            int evenNextIndex = getEvenNextIndex(arr, min, arr[i], valToIndex);
            if(oddNextIndex!=-1){
                dp[i][1] = dp[arr[oddNextIndex]][0];
            }
            if(evenNextIndex!=-1){
                dp[i][0] = dp[arr[evenNextIndex]][1];
            }
            valToIndex[arr[i]] = i;
            max = Math.max(max, arr[i]);
            min = Math.min(min, arr[i]);
            if(dp[i][1]) ans++;
        }
        return ans;
    }

    public int totalFruit(int[] fruits) {
        if(fruits.length ==0) return 0;
        int index1S = -1, index2S = -1;
        int index1E = -1, index2E = -1;
        int type1 = -1, type2=-1;
        int maxCount = 0;
        for(int i=0;i<fruits.length;i++){
            if(type1==fruits[i]){
                index1E = i;
            } else if(type2 == fruits[i]){
                index2E = i;
            } else if(type1==-1){
                index1S = i;
                type1 = fruits[i];
                index1E = i;
            } else if(type2 ==-1){
                index2S = i;
                type2 = fruits[i];
                index2E = i;
            } else{
                maxCount = Math.max(i- index1S, maxCount);
                if(index1E < index2E){
                    index1S = index1E+1;
                    index1E = index2E;
                    type1= type2;
                } else{
                    index1S = index2E+1;
                }

               type2 = fruits[i];
               index2S =i;
               index2E = i;
            }
        }
        maxCount = Math.max(fruits.length- index1S, maxCount);
        return maxCount;
    }

    public int lengthOfLongestSubstring(String s) {
        int left =0, right =0;
        int[] hash = new int[256];
        Arrays.fill(hash, -1);
        int maxLen = 1;
        while (right< s.length()){
            if(hash[s.charAt(right)]>=left){
                maxLen = Math.max(maxLen, right - left);
                left = hash[s.charAt(right)]+1;
            }
            hash[s.charAt(right)] = right++;
        }
        maxLen= Math.max(maxLen, right-left);
        return maxLen;
    }

    public List<List<Integer>> threeSum1(int[] nums) {
        Set<List<Integer>> set = new HashSet<>();
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        for(int i=0;i<nums.length;i++){
            map.put(nums[i], map.getOrDefault(nums[i],0)+1);
        }
        Set<Integer> seen = new HashSet<>();
        for(int i=0;i<n;i++){
            map.put(nums[i], map.get(nums[i])-1);
            if(!seen.contains(nums[i])){
                for(int j =i+1;j<n;j++){
                    if(i==j) continue;
                    map.put(nums[j], map.get(nums[j])-1);
                    if(map.getOrDefault(-nums[i]-nums[j],0)!=0){
                        List<Integer> ans = new ArrayList<>();
                        ans.add(nums[i]);
                        ans.add(nums[j]);
                        ans.add(-nums[i]-nums[j]);
                        Collections.sort(ans);
                        set.add(ans);
                    }
                    map.put(nums[j], map.get(nums[j])+1);
                }
                seen.add(nums[i]);

            }

        }
        return new ArrayList<>(set);
    }

    public void twoSum(int i, int[] nums, List<List<Integer>> ans){
        int lo = i+1, hi = nums.length-1;
        while(lo<hi){
            if(nums[lo]+nums[hi]>-nums[i]){
                hi--;
            } else if(nums[lo]+nums[hi]<-nums[i]){
                lo++;
            } else{
                ans.add(Arrays.asList(nums[i],nums[lo++],nums[hi--]));
                while(lo<hi && nums[lo]==nums[lo-1]){
                    lo++;
                }
            }
        }
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for(int i=0;i<nums.length;i++){
            if(i==0 || nums[i]!=nums[i-1]){
                twoSum(i,nums,ans);
            }
        }
        return ans;
    }


}















