/**
 * @(#)LCIS.java, 3月 09, 2021.
 * <p>
 * Copyright 2021 fenbi.com. All rights reserved.
 * FENBI.COM PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

/**
 * @author yanhaobj
 */
public class LCIS {

    public int findLengthOfLCIS(int[] nums) {
        int length = nums.length;
        if (length <= 1) {
            return length;
        }
        int curMaxLength = 1;
        int maxLength = 1;
        for (int i = 1; i < length; i++) {
            if (nums[i] > nums[i-1]) {
                curMaxLength++;
            } else {
                maxLength = Math.max(curMaxLength, maxLength);
                curMaxLength = 1;
            }
        }
        return maxLength;
    }
}