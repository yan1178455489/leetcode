import java.util.LinkedList;

/**
 * @Auther: worker
 * @Date: 2018/9/26 15:35
 * @Description:
 */
public class BSTIterator {
    LinkedList<Integer> stack;

    public BSTIterator(TreeNode root) {
        stack = new LinkedList<Integer>();
        midtraverse(root,stack);
    }
    private void midtraverse(TreeNode root,LinkedList<Integer> stack){
        if (root==null){
            return;
        }
        if (root.left!=null){
            midtraverse(root.left,stack);
        }
        stack.offerFirst(root.val);
        if (root.right!=null){
            midtraverse(root.right,stack);
        }
    }
    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        if (stack.isEmpty()){
            return false;
        }
        return true;
    }

    /** @return the next smallest number */
    public int next() {
        return stack.pollLast();
    }
}
