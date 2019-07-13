public class two_sum{
	public static void main(String args[]) { 
        int[] nums={2,7,11,15};
        int target = 9;
        two_sum result = new two_sum();
        int[] answer = result.two_sum(nums,target);
        for(int element:answer){System.out.println(element); };
    } 
	public int[] two_sum(int[] nums,int target){
	for (int i=0;i<nums.length;i++){
		for(int j=i+1;j<nums.length;j++){
			if (nums[j]==target-nums[i]){
				return new int[]{i,j};
			}
		}
	}
	throw new IllegalArgumentException("NO two sum solution");}
}