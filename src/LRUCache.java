import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.LinkedList;

/**
 * @Auther: worker
 * @Date: 2018/9/19 11:16
 * @Description:
 */
class LRUCache {
    int capacity;
    HashMap<Integer,Integer> curMap;
    LinkedList<Integer> curList;
    public LRUCache(int capacity) {
        this.capacity=capacity;
        curMap = new HashMap<Integer,Integer>(capacity);
        curList = new LinkedList<Integer>();
    }

    public int get(int key) {
        if (curList.contains(key)){
            int index=curList.indexOf(key);
            curList.remove(index);
            curList.addFirst(key);
            return curMap.get(key);
        }
        return -1;
    }

    public void put(int key, int value) {
        if (!curList.contains(key)){
            if (curList.size()==capacity){
                curList.pollLast();
            }
            curList.addFirst(key);
            if (!curMap.containsKey(key)){
                curMap.put(key,value);
            }else {
                curMap.remove(key);
                curMap.put(key,value);
            }
        } else {
            int index=curList.indexOf(key);
            curList.remove(index);
            curList.addFirst(key);
            curMap.remove(key);
            curMap.put(key,value);
        }
    }
}
