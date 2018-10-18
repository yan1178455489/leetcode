import java.util.LinkedList;

/**
 * @Auther: worker
 * @Date: 2018/9/21 10:02
 * @Description:
 */
public class MinStack {
    /** initialize your data structure here. */
    int min;
    LinkedList<Integer> stack;
    public MinStack() {
        min=Integer.MAX_VALUE;
        stack = new LinkedList<Integer>();
    }

    public void push(int x) {
        if (min>x){
            min=x;
        }
        stack.offerFirst(x);
    }

    public void pop() {
        if (stack.getFirst()==min){
            min=Integer.MAX_VALUE;
            for (int i=1;i<stack.size();i++){
                if (min>stack.get(i)){
                    min=stack.get(i);
                }
            }
        }
        stack.pollFirst();
    }

    public int top() {
        return stack.getFirst();
    }

    public int getMin() {
        return min;
    }
}
