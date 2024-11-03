import { useRef, useEffect, useState, useCallback } from "react";
import igv from "igv";
import ComputeButton from "./ComputeButton";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";

interface IGVbrowserProps {
  options: {
    genome: string;
    locus: string;
    tracks: { name: string; url: string; indexURL: string; format: string }[];
  };
}

function IGVbrowser(props: IGVbrowserProps) {
  const igvContainer = useRef(null);
  const browserRef = useRef(null);

  const [currentLocus, setCurrentLocus] = useState(() => {
    // Initialize currentRegion from options.locus
    if (props.options.locus) {
      return props.options.locus;
    }
    return "";
  });

  const handleLocusChange = useCallback((referenceFrame: any) => {
    const frame = referenceFrame[0];
    console.log(frame);
    if (frame) {
      setCurrentLocus(frame.chr + ":" + frame.start + "-" + frame.end);
    }
  }, []);

  useEffect(() => {
    const initBrowser = async () => {
      if (igvContainer.current && !browserRef.current) {
        try {
          // Add reference sequence track to options
          const updatedOptions = {
            ...props.options,
            tracks: [
              {
                name: "Reference sequence",
                type: "sequence",
                order: -1.0,
              },
              ...(props.options.tracks || []),
            ],
          };

          browserRef.current = await igv.createBrowser(
            igvContainer.current,
            updatedOptions
          );
          console.log("IGV browser created");

          if (browserRef.current !== null) {
            const current: any = browserRef.current;
            console.log(current);
            current.on("locuschange", handleLocusChange);
            const currentLocus = current.currentLoci()[-1];
            if (currentLocus) {
              handleLocusChange(currentLocus);
            }
          }
        } catch (error) {
          console.error("Error creating IGV browser:", error);
        }
      }
    };

    initBrowser();

    return () => {
      if (browserRef.current) {
        const current: any = browserRef.current;
        current.removeAllEventListeners();
        igv.removeAllBrowsers();
        browserRef.current = null;
      }
    };
  }, [props.options, handleLocusChange]);

  return (
    <Card>
      <CardHeader>
        <h1 className="text-3xl font-semibold">LLM DNA compression</h1>
      </CardHeader>
      <CardContent>
        <div ref={igvContainer} />
      </CardContent>
      <CardFooter className="flex justify-center items-center">
        {currentLocus && (
          <div>
            <h3>Current Region: {currentLocus}</h3>
            <ComputeButton locus={currentLocus} />
          </div>
        )}
      </CardFooter>
    </Card>
  );
}

export default IGVbrowser;
