import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Created by worker on 2018/4/25.
 */

public class Solution {
    public static final int[][] go = {
            {0, 1}, {1, 0}
    };

    public static int ans = 0;
    public int[][] go1 = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    int maxSum;
    String[] letters = {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    List<List<String>> res;
    List<String> subres;

    static Interval merge(Interval first, Interval second) {
        int mergeStart = Math.min(first.start, second.start);
        int mergeEnd = Math.max(first.end, second.end);
        Interval merged = new Interval(mergeStart, mergeEnd);
        return span(merged) > span(first) + span(second) ? null : merged;
    }

    public void backtrack(List<List<Integer>> ans,List<Integer> templist,int[] nums,int remain,int start){
        if(remain<0) {
            return;
        }
        if(remain==0) {
            ans.add(new ArrayList<>(templist));
        } else{
            for(int i=start;i<nums.length;i++){
                templist.add(nums[i]);
                backtrack(ans,templist,nums,remain-nums[i],i+1);
                templist.remove(templist.size()-1);
            }
        }
    }

    static int span(Interval i) {
        return i.end - i.start;
    }

    static boolean strictlyBetween(Interval i, int from, int to) {
        return strictlyBetween(i.start, from, to) && strictlyBetween(i.end, from, to);
    }

    static boolean strictlyBetween(int i, int from, int to) {
        return from < i && i < to;
    }

    public void dfs(int[][] maze, int m, int n, int x, int y) {
        for (int i = 0; i <= 1; i++) {
            x += go[i][0];
            y += go[i][1];
            if (x >= m || y >= n) {
                x -= go[i][0];
                y -= go[i][1];
                continue;
            }
            if (x == m - 1 && y == n - 1) {
                ans++;
                return;
            }
            dfs(maze, m, n, x, y);
            x -= go[i][0];
            y -= go[i][1];
        }
    }

    public int uniquePaths(int m, int n) {
        int[][] maze = new int[m][n];

        int x = 0, y = 0;
        //初始化
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                maze[i][j] = 0;
            }
        }
        dfs(maze, m, n, x, y);
        return ans;
    }

    public int minnum(int x, int y, List<List<Integer>> triangle) {
        if (x == triangle.size() - 1) {
            return triangle.get(x).get(y);
        }
        if (minnum(x + 1, y, triangle) > minnum(x + 1, y + 1, triangle)) {
            return triangle.get(x).get(y) + minnum(x + 1, y + 1, triangle);
        }
        return triangle.get(x).get(y) + minnum(x + 1, y, triangle);
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size() - 1;
        List<Integer> bot = triangle.get(n);
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                bot.set(j, triangle.get(i).get(j) + (bot.get(j) > bot.get(j + 1) ? bot.get(j + 1) : bot.get(j)));
            }
        }
        return bot.get(0);
    }

    public void inorder(TreeNode root, List<Integer> res) {
        inorder(root.left, res);
        res.add(root.val);
        inorder(root.right, res);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        inorder(root, res);
        return res;
    }

    public List<TreeNode> generateTrees(int n) {
//        List<TreeNode> ans=new ArrayList<TreeNode>();
        return genTrees(1, n);
    }

    public List<TreeNode> genTrees(int s, int e) {
        List<TreeNode> list = new ArrayList<TreeNode>();
        if (s > e) {
            return null;
        }
        if (s == e) {
            list.add(new TreeNode(s));
            return list;
        }
        for (int i = s; i <= e; i++) {
            List<TreeNode> left = genTrees(s, i - 1);
            List<TreeNode> right = genTrees(i + 1, e);
            for (TreeNode lnode : left) {
                for (TreeNode rnode : right) {
                    TreeNode root = new TreeNode(i);
                    root.left = lnode;
                    root.right = rnode;
                    list.add(root);
                }
            }
        }
        return list;
    }

    public TreeNode build(int s1, int e1, int s2, int e2, int[] preorder, int[] inorder) {
        int inrootIndex = 0;
        for (int i = s2; i <= e2; i++) {
            if (preorder[s1] == inorder[i]) {
                inrootIndex = i;
                break;
            }
        }
        TreeNode root = new TreeNode(preorder[s1]);
        if (inrootIndex != s2) {
            root.left = build(s1 + 1, s1 + (inrootIndex - s2), s2, inrootIndex - 1, preorder, inorder);
        }
        if (inrootIndex != e2) {
            root.right = build(s1 + (inrootIndex - s2) + 1, e1, inrootIndex + 1, e2, preorder, inorder);
        }
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        TreeNode res = build(0, n - 1, 0, n - 1, preorder, inorder);
        return res;
    }

    public List<List<String>> backtrack(int i, int n, int[] row, int[] col, int[] lup, int[] rup, List<List<String>> ans, List<String> subans) {
        if (i >= n) {
            ArrayList<String> tans = new ArrayList<String>(subans);
            ans.add(tans);
            return ans;
        }
        for (int j = 0; j < n; j++) {
            if ((col[j] == 0) && (rup[i + j] == 0) && (lup[i - j + n - 1] == 0)) {
                //若无皇后
                char[] s = new char[n];
                for (int k = 0; k < n; k++) {
                    s[k] = (k == j ? 'Q' : '.');
                }
                String ss = new String(s);
                subans.add(ss);
                //设定为占用
                col[j] = rup[i + j] = lup[i - j + n - 1] = 1;
                backtrack(i + 1, n, row, col, lup, rup, ans, subans);  //循环调用
                subans.remove(ss);
                col[j] = rup[i + j] = lup[i - j + n - 1] = 0;
            }
        }
        return ans;
    }

    public List<List<String>> solveNQueens(int n) {
        int[] row = new int[n];
        int[] col = new int[n];
        int[] lup = new int[n * 2 - 1];//从左下开始
        int[] rup = new int[n * 2 - 1];//从左上开始
        List<List<String>> ans = new ArrayList<List<String>>();
        ArrayList<String> subans = new ArrayList<String>();
        return backtrack(0, n, row, col, lup, rup, ans, subans);
    }

    public int maxsub(int[] nums, int left, int right, int ans) {
        if (left > right) {
            return 0;
        }
        for (int i = left; i <= right; i++) {
            if (nums[i] > 0) {
                ans = nums[i] + maxsub(nums, i + 1, right, ans);
            } else {
                ans = maxsub(nums, i + 1, right, 0) > (nums[i] + maxsub(nums, i + 1, right, ans)) ? maxsub(nums, i + 1, right, 0) : (nums[i] + maxsub(nums, i + 1, right, ans));
            }
        }
        return ans;
    }

    public int maxSubArray(int[] nums) {
        int max = Integer.MIN_VALUE, sum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (sum < 0) {
                sum = nums[i];
            } else {
                sum += nums[i];
            }
            if (sum > max) {
                max = sum;
            }
        }
        return max;
    }

    public int[] maxP(int[] prices, int left, int right) {
        int max = Integer.MIN_VALUE, dif = 0, maxdif = Integer.MIN_VALUE, lefti = left, righti = right;
        for (int i = right; i >= left; i--) {
            if (prices[i] > max) {
                max = prices[i];
                righti = i;
            } else {
                dif = max - prices[i];
                if (dif > maxdif) {
                    maxdif = dif;
                    lefti = i;
                }
            }
        }
        int[] ans = new int[3];
        ans[0] = maxdif;
        ans[1] = lefti;
        ans[2] = righti;
        return ans;
    }

    public int maxProfit1(int[] prices) {
        if (prices.length == 0) {
            return 0;
        }
        int[][] dp = new int[3][prices.length];
        for (int k = 1; k <= 2; k++) {
            int maxT = -prices[0];
            for (int d = 1; d < prices.length; d++) {
                maxT = Math.max(maxT, dp[k - 1][d - 1] - prices[d]);
                dp[k][d] = Math.max(dp[k][d - 1], prices[d] + maxT);
            }
        }
        return dp[2][prices.length - 1];
    }

    List<Integer> backT(int[][] matrix, int m, int n, int start, List<Integer> ans) {
        if (m <= 0 || n <= 0) {
            return null;
        }
        for (int i = start; i <= start + n - 1; i++) {
            ans.add(matrix[start][i]);
        }
        for (int i = start + 1; i <= start + m - 1; i++) {
            ans.add(matrix[i][start + n - 1]);
        }
        for (int i = start + n - 2; i >= start; i--) {
            ans.add(matrix[start + m - 1][i]);
        }
        for (int i = start + m - 2; i >= start + 1; i--) {
            ans.add(matrix[i][start]);
        }
        backT(matrix, m - 2, n - 2, start + 1, ans);
        return ans;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        List<Integer> ans = new ArrayList<Integer>();
        return backT(matrix, m, n, 0, ans);
    }

    public boolean canJ(int[] nums, int pos) {
        if (pos >= (nums.length - 1)) {
            return true;
        }
        for (int step = nums[pos]; step >= 1; step--) {
            pos += step;
            if (canJ(nums, pos)) {
                return true;
            }
            pos -= step;
        }
        return false;
    }

    public List<Interval> merge(List<Interval> intervals) {
        if (intervals.size() <= 1)
            return intervals;
        intervals.sort((i1, i2) -> Integer.compare(i1.start, i2.start));
        List<Interval> res = new ArrayList<Interval>();
        int start = intervals.get(0).start;
        int end = intervals.get(0).end;
        for (Interval interval : intervals) {
            if (end >= interval.start) {
                end = Math.max(interval.end, end);
            } else {
                Interval tmp = new Interval(start, end);
                res.add(tmp);
                start = interval.start;
                end = interval.end;
            }
        }
        res.add(new Interval(start, end));
        return res;
    }

    //    public List<Interval> insert1(List<Interval> intervals, Interval newInterval) {
//        for (Interval interval:intervals){
//            if (newInterval.start>=interval.start&& newInterval.start<=interval.end){
//
//            }
//        }
//    }
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new LinkedList<>();
        int prevEnd = Integer.MIN_VALUE;
        Interval toInsert = newInterval;
        for (Interval i : intervals) {
            if (toInsert != null) {
                Interval mergeResult = merge(i, toInsert);
                if (mergeResult != null) {
                    toInsert = mergeResult;
                    continue;
                }
                if (strictlyBetween(toInsert, prevEnd, i.start)) {
                    result.add(toInsert);
                    toInsert = null;
                }
            }
            result.add(i);
            prevEnd = i.end;
        }
        if (toInsert != null) {
            result.add(toInsert);
        }
        return result;
    }

    public boolean canJump(int[] nums) {
        return canJ(nums, 0);
    }

    public int lengthOfLastWord(String s) {
        if (s.length() == 0) {
            return 0;
        }
        int ans = 0;
        int i = s.length() - 1;
        while (s.charAt(i) == ' ') {
            i--;
        }
        for (; i >= 0; i--) {
            if (s.charAt(i) != ' ') {
                ans++;
            } else {
                break;
            }
        }
        return ans;
    }

    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        if (n == 0) {
            return res;
        }
        int count = 1;
        for (int round = 0; round <= n / 2; round++) {
            System.out.println(round);
            for (int j = round; j <= n - round - 2; j++) {
                res[round][j] = count++;
                System.out.println(res[round][j]);
            }
            for (int j = round; j <= n - round - 2; j++) {
                res[j][n - round - 1] = count++;
            }
            for (int j = n - round - 1; j > round; j--) {
                res[n - round - 1][j] = count++;
            }
            for (int j = n - round - 1; j > round; j--) {
                res[j][round] = count++;
            }
        }
        return res;
    }

    public int factorial(int n) {
        if (n == 1) return 1;
        return n * factorial(n - 1);
    }

    public String getPermutation(int n, int k) {
        int pos = 0;
        List<Integer> numbers = new ArrayList<>();
        int[] factorial = new int[n + 1];
        StringBuilder sb = new StringBuilder();

        // create an array of factorial lookup
        int sum = 1;
        factorial[0] = 1;
        for (int i = 1; i <= n; i++) {
            sum *= i;
            factorial[i] = sum;
        }
        // factorial[] = {1, 1, 2, 6, 24, ... n!}

        // create a list of numbers to get indices
        for (int i = 1; i <= n; i++) {
            numbers.add(i);
        }
        // numbers = {1, 2, 3, 4}

        k--;

        for (int i = 1; i <= n; i++) {
            int index = k / factorial[n - i];
            sb.append(numbers.get(index));
            numbers.remove(index);
            k -= index * factorial[n - i];
        }

        return String.valueOf(sb);
    }

    public ListNode rotateRight(ListNode head, int k) {
        if (head == null || k == 0)
            return head;
        int count = 1;
        ListNode bianli = head;
        while (bianli.next != null) {
            count++;
            bianli = bianli.next;
        }
        System.out.println(count);
        int newhead = 0;
        if (k <= count)
            newhead = count - k + 1;
        else {
            newhead = count - (k % count) + 1;
            System.out.println(k);
        }
        ListNode bianli1 = head;

        if (newhead != 1) {

            ListNode pre = bianli1;
            for (int i = 2; i <= newhead; i++) {
                pre = bianli1;
                bianli1 = bianli1.next;
            }
            bianli.next = head;
            pre.next = null;
        } else {

        }
        return bianli1;
    }

    public int uniquePaths1(int m, int n) {
        if (m == 0 || n == 0)
            return 0;
        if (m == 1 || n == 1) {
            return 1;
        }
        int[][] map = new int[m][n];
        for (int i = 0; i <= m - 1; i++) {
            map[i][0] = 1;
        }
        for (int i = 0; i <= n - 1; i++) {
            map[0][i] = 1;
        }
        for (int i = 1; i <= m - 1; i++) {
            for (int j = 1; j <= n - 1; j++) {
                map[i][j] = map[i - 1][j] + map[i][j - 1];
            }
        }
        return map[m - 1][n - 1];
    }

    public int climbStairs(int n) {
        if (n == 1 || n == 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    //    public int minCostClimbingStairs(int[] cost) {
//        int length=cost.length;
//        int[] dp=new int[length+1];
//        dp[0]=cost[0];
//        dp[1]=cost[1];
//        for (int i=2;i<=length-1;i++){
//            dp[i]=Math.min(dp[i-1],dp[i-2])+cost[i];
//        }
//        return Math.min(dp[length])
//    }
    public int rob(int[] nums, int start, int end) {
        int length = end - start + 1;
        int[] dp = new int[length + 1];
        dp[0] = nums[start];
        dp[1] = Math.max(dp[0], nums[start + 1]);
        for (int i = 2; i <= length - 1; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[start + i], dp[i - 1]);
        }
        return dp[length - 1];
    }

    public boolean searchM(int[] matrix, int target) {
        int start = 0;
        int end = matrix.length - 1;
        int mids = (start + end) / 2;
        while (start < end) {
            if (target == matrix[mids])
                return true;
            else if (target < matrix[mids]) {
                end = mids - 1;
            } else {
                start = mids + 1;
            }
            mids = (start + end) / 2;
        }
        return false;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int start = 0, end = matrix.length - 1;
        int mid = (start + end) / 2;
        while (start < end) {
            if (target == matrix[mid][0])
                return true;
            else if (target < matrix[mid][0]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
            mid = (start + end) / 2;
        }
        if (target > matrix[mid][0]) {
            return searchM(matrix[mid], target);
        } else {
            return searchM(matrix[mid + 1], target);
        }
    }

    public void sortColors(int[] nums) {
        if (nums.length == 0)
            return;
        int count0 = 0, count1 = 0, count2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0)
                count0++;
            else if (nums[i] == 1)
                count1++;
            else count2++;
        }
        for (int i = 0; i < count0; i++) {
            nums[i] = 0;
        }
        for (int i = count0; i < count0 + count1; i++) {
            nums[i] = 1;
        }
        for (int i = count0 + count1; i < count0 + count1 + count2; i++) {
            nums[i] = 2;
        }
    }

    public void bianli(List<Integer> numbers, List<List<Integer>> ans, List<Integer> choosen, int k) {
        if (choosen.size() == k) {
            ans.add(new ArrayList<Integer>(choosen));
            return;
        }
        int a = numbers.get(0);
        numbers.remove(0);
        choosen.add(a);
        bianli(numbers, ans, choosen, k);
        choosen.remove(choosen.size() - 1);
        bianli(numbers, ans, choosen, k);
        numbers.add(0, a);
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> choosen = new ArrayList<Integer>();
        ans.add(new ArrayList<Integer>());
        List<Integer> tmp = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
            tmp.add(nums[i]);
        }
        ans.add(tmp);
        for (int i = 1; i < nums.length; i++) {
            bianli(tmp, ans, choosen, i);
        }
        return ans;
    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> tmp = new ArrayList<Integer>();
        List<Integer> available = new ArrayList<Integer>();
        for (int i = 1; i <= n; i++) {
            available.add(i);
        }
        bianli(available, ans, tmp, k);
        return ans;
    }

    public boolean isExist(char[][] board, char[] wordList, boolean[][] alreadyVisted, int start, int curM, int curN) {
        if (start == wordList.length) return true;
        if (curM < 0 || curM >= board.length || curN < 0 || curN >= board[0].length) return false;
        if (alreadyVisted[curM][curN]) return false;
        if (board[curM][curN] != wordList[start]) return false;
        alreadyVisted[curM][curN] = true;
        boolean result = isExist(board, wordList, alreadyVisted, start + 1, curM + 1, curN) ||
                isExist(board, wordList, alreadyVisted, start + 1, curM, curN + 1) ||
                isExist(board, wordList, alreadyVisted, start + 1, curM - 1, curN) ||
                isExist(board, wordList, alreadyVisted, start + 1, curM, curN - 1);
        if (result) return true;
        alreadyVisted[curM][curN] = false;
        return false;
    }

    public boolean exist(char[][] board, String word) {
        boolean[][] alreadyVisted = new boolean[board.length][board[0].length];
        char[] wordList = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (isExist(board, wordList, alreadyVisted, 0, i, j)) return true;
            }
        }
        return false;
    }

    public void yiwei(int curp, int[] nums, int flag1) {
        for (int i = curp + 1; i < nums.length; i++) {
            nums[i - 1] = nums[i];
        }
        if (flag1 == 1) {
            nums[nums.length - 1] = -1;
        }
    }

    public int removeDuplicates(int[] nums) {
        int len = nums.length;
        if (len <= 2) {
            return len;
        }
        int appearnum = 1;
        int tmp = -1;
        int flag1 = 1;
        for (int i = 0; i < len; i++) {
            if (nums[i] == -1) {
                return i;
            }
            if (nums[i] == tmp) {
                appearnum++;
                if (appearnum > 2) {
                    yiwei(i, nums, flag1);
                    i--;
                    appearnum--;
                    flag1 = 0;
                }
            } else {
                tmp = nums[i];
                appearnum = 1;
            }
        }
        return len;
    }

    public boolean search(int[] nums, int target) {
        int left = 0, right = nums.length, mid = -1;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] == target)
                return true;
            if (nums[mid] < nums[left] || nums[mid] < nums[right]) {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            } else if (nums[mid] > nums[left] || nums[mid] > nums[right]) {
                if (target < nums[mid] && target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
        }
        return false;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) return head;
        ListNode pre = head, cur = head.next;
        while (cur != null) {
            while (cur != null && pre.val == cur.val) {
                cur = cur.next;
            }
            if (pre.next == cur) {
                pre = cur;
            } else {
                pre.next = cur;
                pre = cur;
            }
            cur = cur.next;
        }
        return head;
    }

    public ListNode partition(ListNode head, int x) {
        if (head == null)
            return head;
        ListNode head1 = new ListNode(0);
        ListNode cur1 = head1;
        ListNode head2 = new ListNode(0);
        ListNode cur2 = head2;
        while (head != null) {
            if (head.val < x) {
                cur1.next = new ListNode(head.val);
                cur1 = cur1.next;
            } else {
                cur2.next = new ListNode(head.val);
                cur2 = cur2.next;
            }
            head = head.next;
        }
        cur1.next = head2.next;
        return head1.next;
    }

    public void trackdown(int start, List<Integer> subans, HashSet ans) {
        if (-1 == start) {
            if (!ans.contains(subans))
                ans.add(new ArrayList<Integer>(subans));
            return;
        }
        int tmp = subans.get(start);
        subans.remove(start);
        trackdown(start - 1, subans, ans);
        subans.add(start, tmp);
        trackdown(start - 1, subans, ans);
    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        HashSet ans1 = new HashSet();
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> subans = new ArrayList<Integer>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i++) {
            subans.add(nums[i]);
        }
        trackdown(nums.length - 1, subans, ans1);
        ans.addAll(ans1);
        return ans;
    }

    public void leftmove(int[] nums, int start) {
        for (int i = start; i < nums.length - 1; i++) {
            nums[i] = nums[i + 1];
        }
    }

    public void moveZeroes(int[] nums) {
        if (nums.length < 2) {
            return;
        }
        int flag = 0;
        for (int i = nums.length - 2; i >= 0; i--) {
            if (nums[i] == 0) {
                leftmove(nums, i);
                nums[nums.length - 1] = 0;
            }
        }
    }

    public int numD(int start, String s) {
        if (s.length() - start == 0)
            return 1;
        if (s.charAt(0) == '0')
            return 0;
        if (s.length() - start == 1)
            return 1;
        int ans1 = numD(start + 1, s);
        int pre = Integer.parseInt(s.substring(start, start + 2));
        int ans2 = 0;
        if (pre <= 26) {
            ans2 = numD(start + 2, s);
        }
        return ans1 + ans2;
    }

    public int numDecodings(String s) {
        if (s.charAt(0) == '0')
            return 0;
        if (s.length() == 1)
            return 1;
        int ans1 = numD(1, s);
        int pre = Integer.parseInt(s.substring(0, 2));
        int ans2 = 0;
        if (pre <= 26) {
            ans2 = numD(2, s);
        }
        return ans1 + ans2;
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null || head.next == null)
            return head;
        ListNode pre = new ListNode(-1);
        pre.next = head;
        int step = n - m;
        while (m > 1) {
            pre = pre.next;
            m -= 1;
        }
        ListNode subhead = pre.next;
        ListNode cur = pre.next;
        ListNode tail = cur.next;
        ListNode tmp = null;
        for (int i = 1; i <= step; i++) {
            tmp = tail.next;
            tail.next = cur;
            cur = tail;
            tail = tmp;
        }
        subhead.next = tail;
        pre.next = cur;
        if (m == 1)
            return pre.next;
        return head;
    }

    public int numT(int start, int end) {
        if (start >= end) {
            return 1;
        }
        int ans = 0;
        for (int i = start; i <= end; i++) {
            ans += numT(start, i - 1) * numT(i + 1, end);
        }
        return ans;
    }

    public int numTrees(int n) {
        if (n == 1) {
            return 1;
        }

        return numT(1, n);
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        for (int i = m; i < n + m; i++) {
            nums1[i] = nums2[i - m];
        }
        Arrays.sort(nums1);
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p == null || q == null) {
            return false;
        } else if (p.val == q.val) {
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }
        return false;
    }

    public boolean isS(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p == null || q == null) {
            return false;
        } else if (p.val == q.val) {
            return isS(p.left, q.right) && isS(p.right, q.left);
        }
        return false;
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        TreeNode rightRoot = root.right;
        TreeNode leftRoot = root.left;
        return isS(leftRoot, rightRoot);
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> subans = new ArrayList<Integer>();
        if (root == null) {
            return ans;
        }
        ArrayDeque<TreeNode> que = new ArrayDeque<TreeNode>();
        ArrayDeque<TreeNode> subque = new ArrayDeque<TreeNode>();
        que.offer(root);
        int count = 0;
        while (!que.isEmpty()) {
            while (!que.isEmpty()) {
                TreeNode cur = que.poll();
                subans.add(cur.val);
                if (count % 2 == 0) {
                    if (cur.left != null) {
                        subque.offerFirst(cur.left);
                    }
                    if (cur.right != null) {
                        subque.offerFirst(cur.right);
                    }
                } else {
                    if (cur.right != null) {
                        subque.offerFirst(cur.right);
                    }
                    if (cur.left != null) {
                        subque.offerFirst(cur.left);
                    }
                }
            }
            count += 1;
            ans.add(new ArrayList<Integer>(subans));
            subans.clear();
            que.addAll(new ArrayDeque<TreeNode>(subque));
            subque.clear();
        }
        return ans;
    }

    public boolean isValid(String s) {
        if (s.length() % 2 == 1) {
            return false;
        }
        HashSet<Character> set = new HashSet<Character>();
        set.add('(');
        set.add('{');
        set.add('[');
        HashMap<Character, Character> map = new HashMap<Character, Character>();
        map.put(')', '(');
        map.put(']', '[');
        map.put('}', '{');
        ArrayDeque<Character> stack = new ArrayDeque<Character>();
        for (int i = 0; i < s.length(); i++) {
            if (set.contains(s.charAt(i))) {
                stack.offerFirst(s.charAt(i));
            } else {
                if (!stack.isEmpty()) {
                    char ch = stack.pollFirst();
                    if (ch != map.get(s.charAt(i))) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftmax = maxDepth(root.left);
        int rightmax = maxDepth(root.right);
        return leftmax > rightmax ? (1 + leftmax) : (1 + rightmax);
    }

    public TreeNode helper(int[] nums, int low, int high) {
        if (low > high) {
            return null;
        }
        int mid = (high + low) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = helper(nums, low, mid - 1);
        node.right = helper(nums, mid + 1, high);
        return node;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0)
            return null;
        return helper(nums, 0, nums.length - 1);
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null)
            return true;
        if (Math.abs(maxDepth(root.left) - maxDepth(root.right)) > 1) {
            return isBalanced(root.left) && isBalanced(root.right);
        } else {
            return false;
        }
    }

    public int minDepth(TreeNode root) {
        if (root.left == null && root.right == null) {
            return 1;
        }
        int leftmin = root.left == null ? Integer.MAX_VALUE : minDepth(root.left);
        int rightmin = root.right == null ? Integer.MAX_VALUE : minDepth(root.right);
        return leftmin < rightmin ? (1 + leftmin) : (1 + rightmin);
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && root.val == sum) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    public void pathSum(List<List<Integer>> ans, List<Integer> subans, TreeNode root, int sum) {
        if (root == null) {
            return;
        }
        subans.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            ans.add(new ArrayList<Integer>(subans));
            return;
        }
        pathSum(ans, subans, root.left, sum - root.val);
        pathSum(ans, subans, root.right, sum - root.val);
        subans.remove(subans.size() - 1);
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null || root.val > sum) {
            return null;
        }
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        List<Integer> subans = new ArrayList<Integer>();
        pathSum(ans, subans, root, sum);
        return ans;
    }

    public void findDistinct(String s, String t, int num) {
        if (t.length() == 0) {
            num += 1;
            return;
        }
        for (int i = 0; i < (s.length() - t.length() + 1); i++) {
            if (s.charAt(i) == t.charAt(0)) {
                findDistinct(s.substring(i + 1), t.substring(1), num);
            }
        }
    }

    public int numDistinct(String s, String t) {
        int[][] mem = new int[t.length() + 1][s.length()];

        for (int i = 0; i <= s.length(); i++) {
            mem[0][i] = 1;
        }
        for (int i = 1; i <= t.length(); i++) {
            mem[i][0] = 0;
        }
        for (int i = 1; i <= t.length(); i++) {
            for (int j = 1; j <= s.length(); j++) {
                if (s.charAt(j - 1) == t.charAt(i - 1)) {
                    mem[i][j] = mem[i - 1][j - 1] + mem[i][j - 1];
                } else {
                    mem[i][j] = mem[i][j - 1];
                }
            }
        }
        return mem[t.length()][s.length()];
    }

    public int findMaxPath(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = findMaxPath(root.left) > 0 ? findMaxPath(root.left) : 0;
        int right = findMaxPath(root.right) > 0 ? findMaxPath(root.right) : 0;
        maxSum = Math.max(maxSum, left + right + root.val);
        return Math.max(left, right) + root.val;
    }

    public int maxPathSum(TreeNode root) {
        maxSum = Integer.MIN_VALUE;
        findMaxPath(root);
        return maxSum;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode ansHead = new ListNode(0);
        ListNode head = ansHead;
        int s = 0, c = 0;
        while (l1 != null && l2 != null) {
            if ((l1.val + l2.val) > 9) {
                s = l1.val + l2.val + c - 10;
                c = 1;
            } else {
                s = l1.val + l2.val + c;
                c = 0;
            }
            ansHead.next = new ListNode(s);
            ansHead = ansHead.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        while (l1 != null) {
            if ((l1.val + c) > 9) {
                s = l1.val + c - 10;
                c = 1;
            } else {
                s = l1.val + c;
                c = 0;
            }
            ansHead.next = new ListNode(s);
            ansHead = ansHead.next;
            l1 = l1.next;
        }
        while (l2 != null) {
            if ((l2.val + c) > 9) {
                s = l2.val + c - 10;
                c = 1;
            } else {
                s = l2.val + c;
                c = 0;
            }
            ansHead.next = new ListNode(s);
            ansHead = ansHead.next;
            l2 = l2.next;
        }
        if (c != 0) {
            ansHead.next = new ListNode(c);
        }
        return head.next;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s.length() <= 1) {
            return s.length();
        }
        HashSet<Character> set = new HashSet<Character>();
        int longest = -1;
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (!set.contains(s.charAt(i))) {
                set.add(s.charAt(i));
                count += 1;
            } else {
                if (longest < count) {
                    longest = count;
                }
                set.clear();
                count = 0;
                i -= 1;
            }
        }
        return longest;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length > nums2.length ? nums1.length : nums2.length;
        int n = nums1.length <= nums2.length ? nums1.length : nums2.length;
        int[] A = nums1.length > nums2.length ? nums1 : nums2;
        int[] B = nums1.length <= nums2.length ? nums1 : nums2;
        int imin = 0, imax = m;
        int i = 0, j = 0, max_left = 0, max_right = 0;
        while (imin <= imax) {
            i = (imin + imax) / 2;
            j = (m + n + 1) / 2 - i;
            if (i > 0 && A[i - 1] > B[j]) {
                imax = i - 1;
            } else if (i < m && B[j - 1] > A[i]) {
                imin = i + 1;
            } else {
                if (i == 0) {
                    max_left = B[j - 1];
                } else if (j == 0) {
                    max_left = A[i - 1];
                } else {
                    max_left = Math.max(A[i - 1], B[j - 1]);
                }
                if (i == m - 1) {
                    max_right = B[j];
                } else if (j == n - 1) {
                    max_right = A[i];
                } else {
                    max_right = Math.max(B[j], A[i]);
                }
                if ((m + n) % 2 == 1) {
                    return max_left;
                } else {
                    return (max_left + max_right) / 2.0;
                }
            }
        }
        return 0;
    }

    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, maxA = -1;
        while (left < right) {
            if (height[left] < height[right]) {
                int curA = (right - left) * height[left];
                if (maxA < curA) {
                    maxA = curA;
                }
                left += 1;
            } else {
                int curA = (right - left) * height[right];
                if (maxA < curA) {
                    maxA = curA;
                }
                right -= 1;
            }
        }
        return maxA;
    }

    public List<List<Integer>> threeSum(int[] nums, List<List<Integer>> ans, int target, int start) {
        if (nums.length < 3) {
            return ans;
        }
        for (int i = start; i < nums.length - 2; i++) {
            int remain = target - nums[i], low = i + 1, high = nums.length - 1;
            while (low < high) {
                if ((nums[low] + nums[high]) == remain) {
                    ans.add(Arrays.asList(nums[start - 1], nums[i], nums[low], nums[high]));
                    while (low < high && nums[low] == nums[low + 1]) low++;
                    while (low < high && nums[high] == nums[high - 1]) high--;
                    low++;
                    high--;
                } else if ((nums[low] + nums[high]) < remain) {
                    low++;
                } else {
                    high--;
                }
            }
        }
        HashSet h = new HashSet(ans);
        ans.clear();
        ans.addAll(h);
        return ans;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; i++) {
            int remain = -nums[i];
            threeSum(nums, ans, remain, i + 1);
        }
        return ans;
    }

    public List<String> letterCombinations(String digits) {
        List<String> ans = new ArrayList<String>();
        if (digits.length() == 1) {
            StringBuilder stringBuilder = new StringBuilder();
            int no = digits.charAt(0) - '0';
            String start = letters[no];
            for (int i = 0; i < start.length(); i++) {
                ans.add(start.substring(i, i + 1));
            }
        }
        if (digits.length() == 0) {
            return ans;
        }
        StringBuilder stringBuilder = new StringBuilder();
        int no = digits.charAt(0) - '0';
        String start = letters[no];
        for (int i = 0; i < start.length(); i++) {
            stringBuilder.append(start.charAt(i));
            List<String> subans = letterCombinations(digits.substring(1));
            for (String str : subans) {
                stringBuilder.append(str);
                ans.add(stringBuilder.toString());
                stringBuilder.delete(1, stringBuilder.length());
            }
            stringBuilder.delete(0, stringBuilder.length());
        }
        return ans;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return head;
        }
        ListNode start = new ListNode(0);
        ListNode fast = start, slow = start;
        fast.next = head;
        for (int i = 1; i <= n + 1; i++) {
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }

    public ListNode oddEvenList(ListNode head) {
        ListNode odd = head;
        if (head.next == null) {
            return head;
        }
        ListNode evenhead = head.next;
        ListNode even = head.next;
        while (even.next != null) {
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            if (odd.next != null) {
                even = even.next;
            } else {
                break;
            }
        }
        odd.next = evenhead;
        return head;
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> beginSet = new HashSet<String>();
        HashSet<String> endSet = new HashSet<String>();
        HashSet<String> visited = new HashSet<String>();
        beginSet.add(beginWord);
        if (!wordList.contains(endWord)) {
            return 0;
        }
        endSet.add(endWord);
        int len = 1;
        int strlen = beginWord.length();
        while (!beginSet.isEmpty() && !endSet.isEmpty()) {
            if (beginSet.size() > endSet.size()) {
                HashSet<String> set = endSet;
                endSet = beginSet;
                beginSet = set;
            }
            HashSet<String> temp = new HashSet<String>();
            for (String word : beginSet) {
                char[] charArray = word.toCharArray();
                for (int i = 0; i < strlen; i++) {
                    char old = charArray[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        charArray[i] = c;
                        String target = String.valueOf(charArray);
                        if (endSet.contains(target)) {
                            return len + 1;
                        }
                        if (!visited.contains(target) && wordList.contains(target)) {
                            visited.add(target);
                            temp.add(target);
                        }
                    }
                    charArray[i] = old;
                }
            }
            len++;
            beginSet = temp;
        }
        return 0;
    }

    public int longestConsecutive(int[] nums) {
        HashSet<Integer> set = new HashSet<Integer>();
        int maxlen = 0;
        int len = 1;
        for (int i = 0; i < nums.length; i++) {
            set.add(nums[i]);
        }
        for (int i = 0; i < nums.length; i++) {
            if (!set.contains(nums[i] - 1)) {
                int tmp = nums[i];
                while (tmp < nums.length && set.contains(tmp + 1)) {
                    len++;
                    tmp++;
                }
                if (len > maxlen) {
                    maxlen = len;
                    len = 1;
                }
            }
        }
        return maxlen;
    }

    public int sumNumber(TreeNode root, int lastSum) {
        if (root.right == null && root.left == null) {
            return lastSum * 10 + root.val;
        }
        int left = 0, right = 0;
        if (root.left != null) {
            left = sumNumber(root.left, lastSum * 10 + root.val);
        }
        if (root.right != null) {
            right = sumNumber(root.right, lastSum * 10 + root.val);
        }
        return left + right;
    }

    public int sumNumbers(TreeNode root) {
        if (root.right == null && root.left == null) {
            return root.val;
        }
        int left = 0, right = 0;
        if (root.left != null) {
            left = sumNumber(root.left, root.val);
        }
        if (root.right != null) {
            right = sumNumber(root.right, root.val);
        }
        return left + right;
    }

    public void dfs1(char[][] board, char[][] copy, int xaxis, int yaxis, boolean[][] visited) {
        for (int i = 0; i < 4; i++) {
            int nowx = xaxis + go1[0][0];
            int nowy = xaxis + go1[0][1];
            if (nowx >= board.length || nowx < 0 || nowy >= board[0].length || nowy < 0) {
                continue;
            }
            if (visited[nowx][nowy]) {
                continue;
            }
            if (board[nowx][nowy] == 'O') {
                visited[nowx][nowy] = true;
                copy[nowx][nowy] = 'O';
                dfs1(board, copy, nowx, nowy, visited);
            }
        }
    }

    public void solve(char[][] board) {
        if (board == null) {
            return;
        }
        char[][] copy = new char[board.length][board[0].length];
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                copy[i][j] = 'X';
            }
        }
        for (int i = 0; i < board.length; i++) {
            if (board[i][0] == 'O') {
                copy[i][0] = 'O';
                dfs1(board, copy, i, 0, visited);
            }
            if (board[i][board[0].length - 1] == 'O') {
                copy[i][board[0].length - 1] = 'O';
                dfs1(board, copy, i, board[0].length - 1, visited);
            }
        }
        for (int j = 1; j < board[0].length - 1; j++) {
            if (board[0][j] == 'O') {
                copy[0][j] = 'O';
                dfs1(board, copy, 0, j, visited);
            }
            if (board[board.length - 1][j] == 'O') {
                copy[board.length - 1][j] = 'O';
                dfs1(board, copy, board.length - 1, j, visited);
            }
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                board[i][j] = copy[i][j];
            }
        }
    }

    public boolean isPalindrome(String s, int start, int end) {
        if (start == end) {
            return true;
        }
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }

    public void partitionBacktrack(String s, int start) {
        if (subres.size() > 0 && start >= s.length()) {
            res.add(new ArrayList<String>(subres));
            return;
        }
        for (int i = start; i < s.length(); i++) {
            if (isPalindrome(s, start, i)) {
                subres.add(s.substring(start, i + 1));
                partitionBacktrack(s, i + 1);
                subres.remove(subres.size() - 1);
            }
        }
    }

    public List<List<String>> partition(String s) {
        res = new ArrayList<List<String>>();
        subres = new ArrayList<String>();
        partitionBacktrack(s, 0);
        return res;
    }

    public int candy(int[] ratings) {
        int[] candys = new int[ratings.length];
        for (int i = 0; i < ratings.length; i++) {
            candys[i] = 1;
        }
        int ans = 0;
        for (int i = 0; i < ratings.length - 1; i++) {
            if (ratings[i + 1] > ratings[i]) {
                if (candys[i + 1] <= candys[i]) {
                    candys[i + 1] = candys[i] + 1;
                }
            } else if (ratings[i + 1] < ratings[i]) {
                if (candys[i + 1] >= ratings[i]) {
                    candys[i] = candys[i + 1] + 1;
                }
            }
        }
        for (int i = ratings.length - 1; i > 0; i--) {
            if (ratings[i - 1] > ratings[i]) {
                if (candys[i - 1] <= candys[i]) {
                    candys[i - 1] = candys[i] + 1;
                }
            } else if (ratings[i - 1] < ratings[i]) {
                if (candys[i - 1] >= ratings[i]) {
                    candys[i] = candys[i - 1] + 1;
                }
            }
        }
        for (int i = 0; i < ratings.length; i++) {
            ans += candys[i];
        }
        return ans;
    }

    public int[] singleNumber(int[] nums) {
        ArrayList<Integer> set = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (!set.contains(nums[i])) {
                set.add(nums[i]);
            } else {
                set.remove((Object) nums[i]);
            }
        }
        int[] tmp = new int[set.size()];
        for (int i = 0; i < set.size(); i++) {
            tmp[i] = set.get(i);
        }
        return tmp;
    }

    public boolean isinlist(int start, String s, List<String> wordDict) {
        if (start >= s.length()) {
            return true;
        }
        for (String str : wordDict) {
            if ((start + str.length()) <= s.length() && str.equals(s.substring(start, start + str.length()))) {
                int newstart = start + str.length();
                if (isinlist(newstart, s, wordDict)) {
                    return true;
                }
            }
        }
        return false;
    }

    public List<String> dfs(String s, List<String> wordDict, Hashtable<String, LinkedList<String>> hashtable) {
        if (hashtable.containsKey(s)) {
            return hashtable.get(s);
        }
        LinkedList<String> res = new LinkedList<String>();
        if (s.length() == 0) {
            res.add("");
            return res;
        }
        for (String str : wordDict) {
            if (s.startsWith(str)) {
                List<String> tmp = dfs(s.substring(str.length()), wordDict, hashtable);
                String begin = str;
                for (String string : tmp) {
                    begin = begin + string == "" ? string : (" " + string);
                }
                res.add(begin);
            }
        }
        hashtable.put(s, res);
        if (res.isEmpty()) {
            res.add("");
        }
        return res;
    }

    public List<String> wordBreak(String s, List<String> wordDict) {
        Hashtable<String, LinkedList<String>> hashtable = new Hashtable<String, LinkedList<String>>();
        return dfs(s, wordDict, hashtable);
    }

    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode cur = head.next;
        HashSet<ListNode> set = new HashSet<ListNode>();
        set.add(head);
        while (cur != null) {
            if (set.contains(cur)) {
                return true;
            }
            set.add(cur);
            cur = cur.next;
        }
        return false;
    }

    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode cur = head.next;
        HashSet<ListNode> set = new HashSet<ListNode>();
        set.add(head);
        while (cur != null) {
            if (set.contains(cur)) {
                return cur;
            }
            set.add(cur);
            cur = cur.next;
        }
        return null;
    }

    public void postorder(TreeNode root, ArrayList<Integer> res) {
        if (root == null) {
            return;
        }
        postorder(root.left, res);
        postorder(root.right, res);
        res.add(root.val);
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        if (root == null) {
            return res;
        }
        postorder(root.left, res);
        postorder(root.right, res);
        res.add(root.val);
        return res;
    }

    public ListNode merges(ListNode left, ListNode right) {
        ListNode fakehead = new ListNode(0);
        ListNode p = fakehead;
        while (left != null && right != null) {
            if (left.val < right.val) {
                p.next = left;
                left = left.next;
            } else {
                p.next = right;
                right = right.next;
            }
            p = p.next;
        }
        if (left != null) {
            p.next = left;
        }
        if (right != null) {
            p.next = right;
        }
        return fakehead.next;
    }

    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        //把链表分两半
        ListNode pre = null, mid = head, last = head;
        while (last != null && last.next != null) {
            pre = mid;
            mid = mid.next;
            last = last.next.next;
        }
        pre.next = null;
        //分别排序
        ListNode left = sortList(head);
        ListNode right = sortList(mid);
        //归并
        return merges(left, right);
    }

    public int evalRPN(String[] tokens) {
        int[] nums = new int[tokens.length];
        for (int i = 2; i < tokens.length; i++) {
            if ("*".equals(tokens[i]) || "/".equals(tokens[i]) || "+".equals(tokens[i]) || "-".equals(tokens[i])) {
                int j = i - 1, k = i - 2;
                while (tokens[k] == ";") {
                    k--;
                }
                if (tokens[i] == "+") {
                    int a = Integer.parseInt(tokens[k]);
                    int b = Integer.parseInt(tokens[j]);
                    a += b;
                    tokens[i] = String.valueOf(a);
                    tokens[j] = ";";
                    tokens[k] = ";";
                } else if (tokens[i] == "-") {
                    int a = Integer.parseInt(tokens[k]);
                    int b = Integer.parseInt(tokens[j]);
                    a -= b;
                    tokens[i] = String.valueOf(a);
                    tokens[j] = ";";
                    tokens[k] = ";";
                } else if (tokens[i] == "*") {
                    int a = Integer.parseInt(tokens[k]);
                    int b = Integer.parseInt(tokens[j]);
                    a *= b;
                    tokens[i] = String.valueOf(a);
                    tokens[j] = ";";
                    tokens[k] = ";";
                } else if (tokens[i] == "/") {
                    int a = Integer.parseInt(tokens[k]);
                    int b = Integer.parseInt(tokens[j]);
                    a /= b;
                    tokens[i] = String.valueOf(a);
                    tokens[j] = ";";
                    tokens[k] = ";";
                }
            }
        }
        return Integer.parseInt(tokens[tokens.length - 1]);
    }

    public String reverseWords(String s) {
        String[] ans = s.split(" ");
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = ans.length - 1; i >= 0; i++) {
            if (!"".equals(ans[i])) {
                stringBuilder.append(ans[i]);
                stringBuilder.append(" ");
            }
        }
        int end = 0;
        for (end = stringBuilder.length() - 1; end >= 0 && stringBuilder.charAt(end) == ' '; end--) {
        }
        return stringBuilder.substring(0, end + 1);
    }

    public int maxProduct(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int maxproduct = Integer.MIN_VALUE;
        for (int i = 0; i <= nums.length - 1; i++) {
            int cur = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                if (cur > maxproduct) {
                    maxproduct = cur;
                }
                cur *= nums[j];
            }
            if (cur > maxproduct) {
                maxproduct = cur;
            }
        }
        return maxproduct;
    }

    public int findMin(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i + 1] < nums[i]) {
                return nums[i + 1];
            }
        }
        return nums[0];
    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        HashSet<ListNode> hashSet = new HashSet<ListNode>();
        while (headA != null) {
            hashSet.add(headA);
            headA = headA.next;
        }
        while (headB != null) {
            if (hashSet.contains(headB)) {
                return headB;
            }
            headB = headB.next;
        }
        return null;
    }

    public int maximumGap(int[] nums) {
        if (nums.length <= 1) {
            return 0;
        }
        Arrays.sort(nums);
        int maxgap = Integer.MIN_VALUE;
        for (int i = 0; i < nums.length - 1; i++) {
            int gap = nums[i + 1] - nums[i];
            if (maxgap < gap) {
                maxgap = gap;
            }
        }
        return maxgap;
    }

    public int compareVersion(String version1, String version2) {
        String[] a1 = version1.split("\\.");
        String[] a2 = version2.split("\\.");
        int minlen = Math.min(a1.length, a2.length), i = 0;
        for (i = 0; i < minlen; i++) {
            if (Integer.parseInt(a1[i]) < Integer.parseInt(a2[i])) {
                return -1;
            }
            if (Integer.parseInt(a1[i]) > Integer.parseInt(a2[i])) {
                return 1;
            }
        }
        if (a1.length == a2.length) {
            return 0;
        }
        if (a1.length > a2.length) {
            for (; i < a1.length; i++) {
                if (Integer.parseInt(a1[i]) != 0) {
                    return 1;
                }
            }
        }
        if (a1.length < a2.length) {
            for (; i < a2.length; i++) {
                if (Integer.parseInt(a2[i]) != 0) {
                    return -1;
                }
            }
        }
        return 0;
    }

    public int[] twoSum(int[] numbers, int target) {
        int[] ans = new int[2];
        int left = 0, right = numbers[numbers.length - 1];
        while (left < right) {
            if (numbers[left] + numbers[right] == target) {
                ans[0] = left + 1;
                ans[0] = right + 1;
                return ans;
            }
            if (numbers[left] + numbers[right] < target) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
    }

    public void mid(TreeNode root, ArrayList<Integer> nums) {
        if (root == null) {
            return;
        }
        if (root.left != null) {
            mid(root.left, nums);
        }
        nums.add(root.val);
        if (root.right != null) {
            mid(root.right, nums);
        }
    }

    public boolean findTarget(TreeNode root, int k) {
        if (root == null) {
            return false;
        }
        ArrayList<Integer> nums = new ArrayList<Integer>();
        mid(root, nums);
        int left = 0, right = nums.size() - 1;
        while (left < right) {
            if (nums.get(left) + nums.get(right) == k) {
                return true;
            }
            if (nums.get(left) + nums.get(right) < k) {
                left++;
            } else {
                right--;
            }
        }
        return false;
    }

    public String convertToTitle(int n) {
        StringBuilder stringBuilder = new StringBuilder();
        LinkedList<Character> linkedList = new LinkedList<Character>();
        int cur = 0;
        int power = 0;
        while (n > 0) {
            if (n % 26 != 0) {
                cur = n % 26;
            } else {
                cur = 26;
            }
            char tmp = (char) ('A' + cur - 1);
            linkedList.offerFirst(tmp);
            n = (n - cur * (int) Math.pow(26, power)) / 26;
            power++;
        }
        while (!linkedList.isEmpty()) {
            stringBuilder.append(linkedList.pollFirst());
        }
        return stringBuilder.toString();
    }

    public int majorityElement1(int[] nums) {
        HashMap<Integer, Integer> hashMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < nums.length; i++) {
            if (hashMap.containsKey(nums[i])) {
                if (hashMap.get(nums[i]) + 1 > nums.length / 2) {
                    return nums[i];
                }
                hashMap.replace(nums[i], hashMap.get(nums[i]) + 1);

            } else {
                hashMap.put(nums[i], 1);
            }
        }
        return 0;
    }

    public List<Integer> majorityElement(int[] nums) {
        List<Integer> ans = new ArrayList<Integer>();
        if (nums.length == 0) {
            return ans;
        }
        int major1 = nums[0], major2 = nums[0], count1 = 0, count2 = 0, len = nums.length;
        for (int i = 0; i < nums.length; i++) {
            if (major1 == nums[i]) {
                count1++;
            } else if (major2 == nums[i]) {
                count2++;
            } else if (count1 == 0) {
                major1 = nums[i];
                count1 = 1;
            } else if (count2 == 0) {
                major2 = nums[i];
                count2 = 1;
            } else {
                count1--;
                count2--;
            }
        }
        count1 = 0;
        count2 = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == major1) {
                count1++;
            } else if (nums[i] == major2) {
                count2++;
            }
        }
        if (count1 > len / 3) {
            ans.add(major1);
        }
        if (count2 > len / 3) {
            ans.add(major2);
        }
        return ans;
    }

    public int trailingZeroes(int n) {
        int count = 0;
        while (n >= 5) {
            count += n / 5;
            n /= 5;
        }
        return count;
    }

    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        int n = dungeon[0].length;
        int[][] dp = new int[m + 1][n + 1];
        for (int i = m - 1; i >= 0; i--) {
            dp[i][n] = Integer.MAX_VALUE;
        }
        for (int i = n - 1; i >= 0; i--) {
            dp[m][i] = Integer.MAX_VALUE;
        }
        dp[m][n - 1] = 1;
        dp[m - 1][n] = 1;
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j];
                dp[i][j] = dp[i][j] <= 0 ? 1 : dp[i][j];
            }
        }
        return dp[0][0];
    }

    public String largestNumber(int[] nums) {
        if (nums.length == 1) {
            return String.valueOf(nums[0]);
        }
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            strings[i] = String.valueOf(nums[i]);
        }
        Comparator<String> comparator = new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String str1 = o1 + o2;
                String str2 = o2 + o1;
                return str2.compareTo(str1);
            }
        };
        Arrays.sort(strings, comparator);
        if (strings[0].equals("0")) {
            return "0";
        }
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < strings.length; i++) {
            stringBuilder.append(strings[i]);
        }
        return stringBuilder.toString();
    }

    public int quicksolve(int[] prices) {
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i] < prices[i + 1]) {
                profit += (prices[i + 1] - prices[i]);
            }
        }
        return profit;
    }

    public int maxProfit(int k, int[] prices) {
        if (prices.length <= 1) {
            return 0;
        }
        if (k >= prices.length / 2) {
            return quicksolve(prices);
        }
        int[][] dp = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            int maxT = -prices[0];
            for (int j = 1; j < prices.length; j++) {
                maxT = Math.max(maxT, dp[i - 1][j - 1] - prices[j]);
                dp[i][j] = Math.max(dp[i][j - 1], prices[j] + maxT);
            }
        }
        return dp[k][prices.length - 1];
    }

    public List<String> findRepeatedDnaSequences(String s) {
        int len = 10;
        List<String> ans = new ArrayList<String>();
        if (s.length() <= 10) {
            return ans;
        }
        HashSet<String> set = new HashSet<String>();
        for (int i = 0; i < s.length() - 9; i++) {
            String tmp = s.substring(i, i + 10);
            if (!set.contains(tmp)) {
                set.add(tmp);
            } else {
                if (!ans.contains(tmp)) {
                    ans.add(tmp);
                }
            }
        }
        return ans;
    }

    public void rotate(int[] nums, int k) {
        for (int i = 1; i <= k; i++) {
            int tmp = nums[nums.length - 1];
            for (int j = nums.length - 2; j >= 0; j--) {
                nums[j + 1] = nums[j];
            }
            nums[0] = tmp;
        }
    }

    public void reverse(int[] nums) {
        int start = 0;
        int end = nums.length - 1;
        while (start < end) {
            int tmp = nums[end];
            nums[end] = nums[start];
            nums[start] = tmp;
            start++;
            end--;
        }
    }

    public int reverseBits(int n) {
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            ans += n & 1;
            n = n >>> 1;
            if (i < 31) {
                ans = ans << 1;
            }
        }
        return ans;
    }

    public int hammingWeight(int n) {
        int count = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & 1) == 1) {
                count++;
            }
            n = n >>> 1;
        }
        return count;
    }

    public boolean reachingPoints(int sx, int sy, int tx, int ty) {
        while (tx >= sx && ty >= sy) {
            if (tx > ty) {
                if (ty == sy) {
                    return (tx - sx) % ty == 0;
                } else {
                    tx %= ty;
                }
            } else {
                if (tx == sx) {
                    return (ty - sy) % tx == 0;
                } else {
                    ty %= tx;
                }
            }
        }
        return false;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<ListNode>(lists.length, new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                if (o1.val > o2.val) {
                    return 1;
                }
                if (o1.val == o2.val) {
                    return 0;
                }
                return -1;
            }
        });
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        for (int i = 0; i < lists.length; i++) {
            priorityQueue.add(lists[i]);
        }
        while (!priorityQueue.isEmpty()) {
            tail.next = priorityQueue.poll();
            tail = tail.next;

            if (tail.next != null) {
                priorityQueue.add(tail.next);
            }
        }
        return dummy.next;
    }

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode cur = head;
        int count = 0;
        while (cur != null && count < k) {
            cur = cur.next;
            count++;
        }
        if (count == k) {
            cur = reverseKGroup(cur, k);
            while (count-- > 0) {
                ListNode tmp = head.next;
                head.next = cur;
                cur = head;
                head = tmp;
            }
            head = cur;
        }
        return head;
    }

    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> ans = new ArrayList<Integer>();
        int slen = s.length(), num = words.length;
        if (slen == 0 || num == 0) {
            return ans;
        }
        int wlen = words[0].length();
        if (slen < wlen * num) {
            return ans;
        }
        HashMap<String, Integer> hashMap = new HashMap<String, Integer>();
        for (String str : words) {
            if (hashMap.containsKey(str)) {
                hashMap.put(str, hashMap.get(str) + 1);
            } else {
                hashMap.put(str, 1);
            }
        }
        for (int i = 0; i <= slen - (wlen * num); i++) {
            int count = 0;
            HashMap<String, Integer> thashMap = new HashMap<String, Integer>();
            for (int j = i; j <= i + wlen * (num - 1); j += wlen) {
                String str = s.substring(j, j + wlen);
                if (hashMap.containsKey(str)) {
                    thashMap.put(str, thashMap.get(str) + 1);
                    if (thashMap.get(str) <= hashMap.get(str)) {
                        count++;
                    } else {
                        break;
                    }
                    if (count == num) {
                        ans.add(i);
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        return ans;
    }

    public int[] searchRange(int[] nums, int target) {
        int len = nums.length;
        int[] ans = {-1, -1};
        int start = 0, end = len - 1, mid;
        while (start <= end) {
            mid = (start + end) / 2;
            if (nums[mid] == target) {
                int left = mid, right = mid;
                while (left > 0 && nums[left - 1] == nums[left]) left--;
                while (right < len - 1 && nums[right] == nums[right + 1]) right++;
                ans[0] = left;
                ans[1] = right;
                return ans;
            }
            if (target < nums[mid]) {
                end = mid - 1;
            } else {
                start = mid + 1;
            }
        }
        return ans;
    }

    public boolean isValidSudoku(char[][] board) {
        HashSet<Character> hashSet1 = new HashSet<Character>();
        HashSet<Character> hashSet2 = new HashSet<Character>();
        for (int i = 0; i <= 8; i++) {
            for (int j = 0; j <= 8; j++) {
                if (board[i][j] != '.') {
                    if (!hashSet1.contains(board[i][j])) {
                        hashSet1.add(board[i][j]);
                    } else {
                        return false;
                    }
                }
                if (board[j][i] != '.') {
                    if (!hashSet2.contains(board[j][i])) {
                        hashSet2.add(board[j][i]);
                    } else {
                        return false;
                    }
                }
            }
            hashSet1.clear();
            hashSet2.clear();
        }
        for (int row = 0; row <= 2; row++) {
            for (int column = 0; column <= 2; column++) {
                for (int i = 0; i <= 2; i++) {
                    for (int j = 0; j <= 2; j++) {
                        if (board[row * 3 + i][column * 3 + j] != '.') {
                            if (!hashSet1.contains(board[row * 3 + i][column * 3 + j])) {
                                hashSet1.add(board[row * 3 + i][column * 3 + j]);
                            } else {
                                return false;
                            }
                        }
                    }
                }
                hashSet1.clear();
            }
        }
        return true;
    }

    public boolean recursionSudoku(char[][] board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (char k = '1'; k <= '9'; k++) {
                        board[i][j] = k;
                        if (isValidSudoku(board)) {
                            if (recursionSudoku(board)) {
                                return true;
                            }
                        }
                    }
                    board[i][j] = '.';
                }
            }
        }
        return true;
    }

    public void solveSudoku(char[][] board) {
        recursionSudoku(board);
    }

    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }
        String[] dp = new String[n + 1];
        dp[1] = "1";
        for (int i = 2; i <= n; i++) {
            int count = 1;
            StringBuilder stringBuilder = new StringBuilder();
            for (int j = 0; j < dp[i - 1].length(); j++) {
                if (j + 1 < dp[i - 1].length()) {
                    if (dp[i - 1].charAt(j + 1) == dp[i - 1].charAt(j)) {
                        count++;
                    } else {
                        stringBuilder.append(count);
                        stringBuilder.append(dp[i - 1].charAt(j));
                        count = 1;
                    }
                } else {
                    stringBuilder.append(count);
                    stringBuilder.append(dp[i - 1].charAt(j));
                    count = 1;
                }
            }
            dp[i] = stringBuilder.toString();
        }
        return dp[n];
    }

    public void combinationTrack(int[] candidates, List<List<Integer>> ans, List<Integer> tmp, int remain, int start) {
        if (remain == 0) {
            ans.add(new ArrayList<Integer>(tmp));
            return;
        }
        if (remain < 0) {
            return;
        }
        for (int i = start; i < candidates.length; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) i++;
            tmp.add(candidates[i]);
            combinationTrack(candidates, ans, tmp, remain - candidates[i], i + 1);
            tmp.remove(tmp.size() - 1);
        }
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        Arrays.sort(candidates);
        combinationTrack(candidates, ans, new ArrayList<Integer>(), target, 0);
        HashSet<List<Integer>> hashSet = new HashSet<List<Integer>>(ans);
        ans.clear();
        ans.addAll(hashSet);
        return ans;
    }

    public String addBinary(String a, String b) {
        StringBuilder stringBuilder = new StringBuilder();
        int aindex = a.length() - 1, bindex = b.length() - 1, carry = 0, sum = 0;
        while (aindex >= 0 || bindex >= 0) {
            sum = carry;
            if (aindex >= 0) {
                sum += a.charAt(aindex--) - '0';
            }
            if (bindex >= 0) {
                sum += b.charAt(bindex--) - '0';
            }
            stringBuilder.append(sum % 2);
            carry = sum / 2;
        }
        if (carry > 0) {
            stringBuilder.append(carry);
        }
        return stringBuilder.reverse().toString();
    }

    public int firstMissingPositive(int[] nums) {
        if (nums.length == 0) {
            return 1;
        }
        List<Integer> a = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
            a.add(nums[i]);
        }
        for (int i = 1; i <= nums.length + 1; i++) {
            if (!a.contains(i)) {
                return i;
            }
        }
        return 1;
    }

    public int trap(int[] height) {
        int maxh = -1;
        for (int i = 0; i < height.length; i++) {
            if (maxh < height[i]) {
                maxh = height[i];
            }
        }
        int flag = 0, area = 0, width = 0;
        for (int i = 1; i <= maxh; i++) {
            for (int j = 0; j < height.length; j++) {
                if (height[j] >= i && flag == 0) {//找到第1根柱子
                    flag = 1;
                } else if (flag == 1 && height[j] <= i - 1) {
                    width++;
                } else if (flag == 1 && height[j] >= i) {//找到第二根柱子
                    area += width;
                    width = 0;
                }
            }
            width = 0;
            flag = 0;
        }
        return area;
    }

    private Set<List<Integer>> twoSum(int[] nums, int start, int end, int target) {
        Set<List<Integer>> res = new HashSet<>();
        while (start < end) {
            if (nums[start] + nums[end] == target) {
                res.add(Arrays.asList(nums[start], nums[end]));
                start++;
                end--;
            } else if (nums[start] + nums[end] < target) {
                start++;
            } else {
                end--;
            }
        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        int length = nums.length;
        List<List<Integer>> res = new ArrayList<>();
        if (length < 3) {
            return res;
        }
        Arrays.sort(nums);

        for (int i = 0; i < length; i++) {
            if (i > 0 && nums[i] == nums[i-1]) {
                continue;
            }
            Set<List<Integer>> subRes = twoSum(nums, i + 1, length - 1, -nums[i]);
            for (List<Integer> subList : subRes) {
                ArrayList<Integer> newList = new ArrayList<>(subList);
                newList.add(0, nums[i]);
                res.add(newList);
            }
        }
        return res;
    }

    public static void main(String[] args) {
        Solution sol = new Solution();
        List<List<Integer>> lists = sol.threeSum(new int[]{-1, 0, 1});
        System.out.println(lists);
    }
}
