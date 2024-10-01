import { useState } from "react";
import { useEncode } from "../hooks/useEncode";
import { useToast } from "../hooks/use-toast";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

interface ComputeButtonProps {
    locus: string;
}

export function ComputeButton(props: ComputeButtonProps) {
    const [percentageResponse, setPercentageResponse] = useState('');
    const [complexityResponse, setComplexityResponse] = useState('');
    const [LZcomplexityResponse, setLZcomplexityResponse] = useState('');
    const { toast } = useToast();
    const { mutateAsync: encodeSeq, isPending, isError, error } = useEncode();

    const handleClick = async () => {
        console.log("Handling click...");
        setPercentageResponse('')
        setLZcomplexityResponse('')
        setComplexityResponse('')
        try {
            toast({
                title: "Encoding Starting",
                description: "Starting to encode the sequence.",
                variant: "default",
            });
            const response = await encodeSeq(props.locus);
            console.log("Response:", response);
            const data: any = response?.data;
            const percentage = data.percentage.toFixed(3)
            const complexity = data.complexity.toFixed(3)
            const lzComplexity = data.lz_complexity
            
            setPercentageResponse("The sequence was compressed by " + percentage + "%");
            setComplexityResponse("The sequence complexity is " + complexity);
            setLZcomplexityResponse("The Lempel-Ziv complexity is " + lzComplexity);

            // Show success toast
            toast({
                title: "Encoding Complete",
                description: "The sequence has been successfully encoded.",
                variant: "default",
            });
        } catch (error) {
            console.error("Error:", error);
            setPercentageResponse("An error occurred");
            
            // Show error toast
            toast({
                title: "Encoding Failed",
                description: "An error occurred while encoding the sequence.",
                variant: "destructive",
            });
        }
    };

    return (
        <div className="flex flex-col items-center space-y-8 py-4">
            <Card>
                <CardHeader>
                    <CardTitle className="text-xl">Encode</CardTitle>
                    <CardDescription>Use LLM and arithmetic coding to compress the sequence at the current locus.</CardDescription>
                </CardHeader>
                <CardContent>
                    <TooltipProvider>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <button
                                    onClick={handleClick}
                                    disabled={isPending}
                                    className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
                                >
                                    {isPending ? 'Computing...' : 'Compute'}
                                </button>
                            </TooltipTrigger>
                            <TooltipContent>
                                <p>Run LLM and arithmetic encoding to compress the sequence</p>
                            </TooltipContent>
                        </Tooltip>
                    </TooltipProvider>
                    
                </CardContent>
                <CardFooter>
                    {isPending && <p className="text-gray-600">Encoding in progress...</p>}
                    {isError && <p className="text-red-500">Error: {(error as Error).message}</p>}
                    <div className=" w-full flex flex-col justify-center items-center">
                        {percentageResponse && <h3 className="text-lg font-semibold">{percentageResponse}</h3>}
                        {complexityResponse && <h3 className="text-lg font-semibold">{complexityResponse}</h3>}
                        {LZcomplexityResponse && <h3 className="text-lg font-semibold">{LZcomplexityResponse}</h3>}
                    </div>
                </CardFooter>
            </Card>

           
        </div>
    );
}

export default ComputeButton;