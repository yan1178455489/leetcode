/**
 * @(#)KthLargest.java, 3月 09, 2021.
 * <p>
 * Copyright 2021 fenbi.com. All rights reserved.
 * FENBI.COM PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
 */

/**
 * @author yanhaobj
 */
public class KthLargest {

    private void swap(int[] nums, int left, int right) {
        int tmp = nums[left];
        nums[left] = nums[right];
        nums[right] = tmp;
    }

    public int findKthLargest(int[] nums, int k) {
        int length = nums.length;
        int round = 0;
        while (++round <= k) {
            for (int i = 1; i <= length - round; i++) {
                if (nums[i-1] > nums[i]) {
                    swap(nums, i-1, i);
                }
            }
        }
        return nums[length - k];
    }

    public static void main(String[] args) {
        int[] nums = new int[]{3,2,1,5,6,4};
        KthLargest kthLargest = new KthLargest();
        System.out.println(kthLargest.findKthLargest(nums, 2));
    }
}