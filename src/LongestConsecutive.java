/**
 * @(#)LongestConsecutive.java, 3月 29, 2021.
 * <p>
 * Copyright 2021 fenbi.com. All rights reserved.
 * FENBI.COM PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

import java.util.Arrays;

/**
 * @author yanhaobj
 */
public class LongestConsecutive {

    public int longestConsecutive(int[] nums) {
        int length = nums.length;
        if (length <= 1) {
            return length;
        }
        Arrays.sort(nums);
        int longestConsecutive = 1;
        int curLength = 1;
        for (int i = 1; i < length; i++) {
            if (nums[i-1] + 1 == nums[i]) {
                curLength++;
            } else if (nums[i-1] != nums[i]){
                longestConsecutive = Math.max(longestConsecutive, curLength);
                curLength = 1;
            }
        }
        return Math.max(longestConsecutive, curLength);
    }
}