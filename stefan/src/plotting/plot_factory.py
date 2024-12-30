from src.pre_processing.tasks.task import Task
from src.pre_processing.tasks.task_group import TaskGroup
from src.plotting.plotter import *


def PlotPipelineFactory(titleSuffix: str, labelColumn: str = 'Label', show: bool = True, savePath: str = None,
                        figSize: tuple[float, float] = (16, 10), dpi: int = 300, mode: str = 'analysis',
                        columns: dict[str, str] = None) -> TaskGroup:
    valid_modes = ['analysis', 'modeling', 'inference']
    assert mode in valid_modes, 'invalid mode selected'

    if columns is None:
        columns = {'Time': 'Time',
                   'Source': 'Source',
                   'Protocol': 'Protocol',
                   'Channel': 'Channel',
                   'Length Packet': 'Length Packet',
                   'Length Header': 'Length Header',
                   'AD Type': 'AD Type',
                   'Company ID': 'Company ID',
                   'UUID': 'UUID',
                   'PDU': 'PDU',
                   'Length MS Data': 'Length MS Data',
                   'Continuity Type': 'Continuity Type',
                   'Length Service Data': 'Length Service Data',
                   'SmartTag Type': 'SmartTag Type',
                   'Broadcast': 'Broadcast',
                   'Malformed': 'Malformed',
                   'Label': 'Label',
                   'File': 'File'}

    pipeline = TaskGroup("Plot Pipeline", inplace=False, idempotency=True)

    pipeline.add(
        Task("Number of Packets",
             executor=PlotNumberOfPackets(columns,
                                          title='Number of Packets - ' + titleSuffix,
                                          figSize=figSize,
                                          dpi=dpi,
                                          yLabel='Number of Packets',
                                          savePath=savePath,
                                          show=show
                                          )
             )
    )
    if mode in valid_modes[:-1]:
        pipeline.add(
            Task("Boxplot BLE Address Interval",
                 executor=PlotAddressIntervalBoxPlot(columns,
                                                     title='Boxplot of BLE Address Interval - ' + titleSuffix,
                                                     figSize=figSize,
                                                     dpi=dpi,
                                                     yLabel='Minutes',
                                                     savePath=savePath,
                                                     show=show
                                                     )
                 )
        )

    if mode in valid_modes[:-1]:
        pipeline.add(
            Task("Mean BLE Address Interval",
                 executor=PlotAddressIntervalMean(columns,
                                                  title='Average BLE Address Interval - ' + titleSuffix,
                                                  figSize=figSize,
                                                  dpi=dpi,
                                                  yLabel='Minutes',
                                                  savePath=savePath,
                                                  show=show
                                                  )
                 )
        )

    if mode == valid_modes[0]:
        pipeline.add(
            Task("Broadcast",
                 executor=PlotBroadcast(columns,
                                        title="Percentage of Broadcast Packets - " + titleSuffix,
                                        figSize=figSize,
                                        dpi=dpi,
                                        savePath=savePath,
                                        show=show,
                                        yLabel='% of Packets'
                                        )
                 )
        )

    if mode == valid_modes[0]:
        pipeline.add(
            Task("Protocol",
                 executor=StackedBarPlot(columns={'Label': labelColumn, 'Feature': 'Protocol'},
                                         title="Protocol - " + titleSuffix,
                                         figSize=figSize,
                                         dpi=dpi,
                                         savePath=savePath,
                                         show=show,
                                         yLabel='% of Packets'
                                         )
                 )
        )

    pipeline.add(
        Task("Channel",
             executor=StackedBarPlot(columns={'Label': labelColumn, 'Feature': 'Channel'},
                                     title="Channel - " + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='% of Packets'
                                     )
             ),
    )

    pipeline.add(
        Task("Header Length BoxPlot",
             executor=BoxPlot(columns={'Label': labelColumn, 'Feature': 'Length Header'},
                              title='Boxplot of Length of Header - ' + titleSuffix,
                              figSize=figSize,
                              dpi=dpi,
                              savePath=savePath,
                              show=show,
                              yLabel='Bytes'
                              )

             )
    )

    pipeline.add(
        Task("Header Length Mean",
             executor=MeanValuesPlot(columns={'Label': labelColumn, 'Feature': 'Length Header'},
                                     title="Average Length of Header - " + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='Bytes'
                                     )
             )
    )

    pipeline.add(
        Task("Packet Length BoxPlot",
             executor=BoxPlot(columns={'Label': labelColumn, 'Feature': 'Length Packet'},
                              title='Boxplot of Length of Packet - ' + titleSuffix,
                              figSize=figSize,
                              dpi=dpi,
                              savePath=savePath,
                              show=show,
                              yLabel='Bytes'
                              )

             )
    )

    pipeline.add(
        Task("Packet Length Mean",
             executor=MeanValuesPlot(columns={'Label': labelColumn, 'Feature': 'Length Packet'},
                                     title="Average Length of Packet - " + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='Bytes'
                                     )

             )
    )

    pipeline.add(
        Task("MS Data Length BoxPlot",
             executor=BoxPlot(columns={'Label': labelColumn, 'Feature': 'Length MS Data'},
                              title='Boxplot of Length of Manufacturer specific Data - ' + titleSuffix,
                              figSize=figSize,
                              dpi=dpi,
                              savePath=savePath,
                              show=show,
                              yLabel='Bits')
             )
    )

    pipeline.add(
        Task("MS Data Length Mean",
             executor=MeanValuesPlot(columns={'Label': labelColumn, 'Feature': 'Length MS Data'},
                                     title="Average Length of Manufacturer specific Data - " + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='Bits'
                                     )
             )
    )
    pipeline.add(

        Task("Service Data Length BoxPlot",
             executor=BoxPlot(columns={'Label': labelColumn, 'Feature': 'Length Service Data'},
                              title='Boxplot of Length of Service Data - ' + titleSuffix,
                              figSize=figSize,
                              dpi=dpi,
                              savePath=savePath,
                              show=show,
                              yLabel='Bits'
                              )
             )
    )

    pipeline.add(
        Task("Service Data Length Mean",
             executor=MeanValuesPlot(columns={'Label': labelColumn, 'Feature': 'Length Service Data'},
                                     title="Average Length of Service Data - " + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel="Bits"
                                     )
             )
    )

    pipeline.add(
        Task("Company ID",
             executor=StackedBarPlot(columns={'Label': labelColumn, 'Feature': 'Company ID'},
                                     title='Company ID - ' + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='% of Packets'
                                     )
             )
    )

    pipeline.add(

        Task("UUID",
             executor=PlotUUID(columns=columns,
                               title='UUIDs - ' + titleSuffix,
                               figSize=figSize,
                               dpi=dpi,
                               savePath=savePath,
                               show=show,
                               yLabel='Average Count per Packet'
                               )
             )
    )

    pipeline.add(

        Task("Continuity Type",
             executor=StackedBarPlot(columns={'Label': labelColumn, 'Feature': 'Continuity Type'},
                                     title='Continuity Type - ' + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='% of Packets')
             )
    )

    pipeline.add(
        Task("SmartTag Type",
             executor=StackedBarPlot(columns={'Label': labelColumn, 'Feature': 'SmartTag Type'},
                                     title='SmartTag Type - ' + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='% of Packets')
             )
    )
    pipeline.add(
        Task("AD Type",
             executor=PlotAdvertisementType(columns=columns,
                                            title='Advertisement Data Type - ' + titleSuffix,
                                            figSize=figSize,
                                            dpi=dpi,
                                            savePath=savePath,
                                            show=show,
                                            yLabel='% of Packets'
                                            )

             )
    )

    pipeline.add(
        Task("PDU Type",
             executor=StackedBarPlot(columns={'Label': labelColumn, 'Feature': 'PDU'},
                                     title='PDU Type - ' + titleSuffix,
                                     figSize=figSize,
                                     dpi=dpi,
                                     savePath=savePath,
                                     show=show,
                                     yLabel='% of Packets'
                                     )
             )
    )

    if mode == valid_modes[0]:
        pipeline.add(
            Task("Malformed",
                 executor=PlotMalformedPacket(columns=columns,
                                              title='Malformed Packet - ' + titleSuffix,
                                              figSize=figSize,
                                              dpi=dpi,
                                              savePath=savePath,
                                              show=show,
                                              yLabel='% of Packets')
                 )
        )

    pipeline.add(

        Task("Packet Rate Boxplot",
             executor=PlotPacketRateBoxPlot(columns=columns,
                                            title='Boxplot of Packet Rate - ' + titleSuffix,
                                            figSize=figSize,
                                            dpi=dpi,
                                            savePath=savePath,
                                            show=show,
                                            yLabel='Packets / Second'
                                            )
             )
    )

    pipeline.add(
        Task("Packet Rate Mean",
             executor=PlotPacketRateMean(columns=columns,
                                         title='Average Packet Rate - ' + titleSuffix,
                                         figSize=figSize,
                                         dpi=dpi,
                                         savePath=savePath,
                                         show=show,
                                         yLabel='Packets / Second'
                                         )
             )
    )

    if mode == valid_modes[0]:
        pipeline.add(
            Task("Packet Rate Graph",
                 executor=PlotPacketRateGraph(columns=columns,
                                              title='Graph of Packet Rate',
                                              figSize=figSize,
                                              dpi=dpi,
                                              savePath=savePath,
                                              show=show,
                                              yLabel='Packets / Second',
                                              xLabel='Time in Hours'
                                              )
                 )
        )

    return pipeline
