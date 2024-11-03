import IGVbrowser from "./IGVbrowser";
function MainWindow() {
  const options = {
    genome: "hg38",
    locus: "chr22:27,251,965-27,252,084",
    tracks: [
      {
        name: "HG00103",
        url: "https://s3.amazonaws.com/1000genomes/data/HG00103/alignment/HG00103.alt_bwamem_GRCh38DH.20150718.GBR.low_coverage.cram",
        indexURL:
          "https://s3.amazonaws.com/1000genomes/data/HG00103/alignment/HG00103.alt_bwamem_GRCh38DH.20150718.GBR.low_coverage.cram.crai",
        format: "cram",
      },
    ],
  };

  return (
    <main>
      <div className="p-8 m-4">
        <IGVbrowser options={options} />
      </div>
    </main>
  );
}

export default MainWindow;
