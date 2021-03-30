/**
 * @(#)Search.java, 3月 05, 2021.
 * <p>
 * Copyright 2021 fenbi.com. All rights reserved.
 * FENBI.COM PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

/**
 * @author yanhaobj
 */
public class Search {

    private int afterIndex(int originIndex, int k, int length) {
        int afterIndex = originIndex - k;
        return afterIndex < 0 ? afterIndex + length : afterIndex;
    }

    public int search(int[] nums, int target) {
        int length = nums.length;
        if (length == 1) {
            return 0;
        }
        int k = 0;
        while (k + 1 < length && nums[k] < nums[k+1]) {
            k++;
        }
        k = length - k - 1;
        int left = 0;
        int right = length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            int midValue = nums[afterIndex(mid, k, length)];
            if (midValue == target) {
                return afterIndex(mid, k, length);
            } else if (target < midValue) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        int[] nums = new int[]{4,5,6,7,0,1,2};
//        Search search = new Search();
//        search.search(nums, 0);
        System.out.println(2 + 7 - 3 % 7);
    }
}