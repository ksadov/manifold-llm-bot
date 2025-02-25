import requests
from typing import Optional
from enum import Enum
from typing import Optional


class OutcomeType(Enum):
    BINARY = "BINARY"
    FREE_RESPONSE = "FREE_RESPONSE"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    NUMERIC = "NUMERIC"
    PSEUDO_NUMERIC = "PSEUDO_NUMERIC"
    BOUNTIED_QUESTION = "BOUNTIED_QUESTION"
    POLL = "POLL"


class AddAnswersMode(Enum):
    ANYONE = "ANYONE"
    ONLYCREATOR = "ONLY_CREATOR"
    DISABLED = "DISABLED"


class Option:
    text: str
    votes: int


class LiteMarket:
    def __init__(
        self,
        id: str,
        creatorId: str,
        creatorUsername: str,
        creatorName: str,
        createdTime: int,
        question: str,
        url: str,
        outcomeType: OutcomeType,
        mechanism: str,
        totalLiquidity: float,
        volume: float,
        volume24Hours: float,
        isResolved: bool,
        uniqueBettorCount: int,
        creatorAvatarUrl: Optional[str] = None,
        closeTime: Optional[int] = None,
        p: Optional[dict] = None,
        probability: Optional[float] = None,
        pool: Optional[dict] = None,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        isLogScale: Optional[bool] = None,
        resolutionTime: Optional[int] = None,
        resolution: Optional[str] = None,
        resolutionProbability: Optional[float] = None,
        lastUpdatedTime: Optional[int] = None,
        lastBetTime: Optional[int] = None,
        marketTier: Optional[str] = None,
        siblingContractId: Optional[str] = None,
        slug: Optional[str] = None,
        token: Optional[str] = None,
        lastCommentTime: Optional[int] = None,
        resolverId: Optional[str] = None,
    ):
        self.id = id
        self.creatorId = creatorId
        self.creatorUsername = creatorUsername
        self.creatorName = creatorName
        self.creatorAvatarUrl = creatorAvatarUrl
        self.createdTime = createdTime
        self.closeTime = closeTime
        self.question = question
        self.url = url
        self.outcomeType = OutcomeType(outcomeType)
        self.mechanism = mechanism
        self.probability = probability
        self.pool = pool
        self.p = p
        self.totalLiquidity = totalLiquidity
        self.value = value
        self.min = min
        self.max = max
        self.isLogScale = isLogScale
        self.volume = volume
        self.volume24Hours = volume24Hours
        self.isResolved = isResolved
        self.resolutionTime = resolutionTime
        self.resolution = resolution
        self.resolutionProbability = resolutionProbability
        self.uniqueBettorCount = uniqueBettorCount
        self.lastUpdatedTime = lastUpdatedTime
        self.lastBetTime = lastBetTime
        self.marketTier = marketTier
        self.siblingContractId = siblingContractId
        self.slug = slug
        self.token = token
        self.lastCommentTime = lastCommentTime
        self.resolverId = resolverId

    def __repr__(self):
        return f"LiteMarket(id={self.id}, question={self.question})"

    def __str__(self):
        return self.__repr__()


class FullMarket(LiteMarket):
    def __init__(
        self,
        id: str,
        creatorId: str,
        creatorUsername: str,
        creatorName: str,
        createdTime: int,
        question: str,
        url: str,
        outcomeType: OutcomeType,
        mechanism: str,
        totalLiquidity: float,
        volume: float,
        volume24Hours: float,
        isResolved: bool,
        uniqueBettorCount: int,
        description: dict,
        textDescription: str,
        creatorAvatarUrl: Optional[str] = None,
        closeTime: Optional[int] = None,
        p: Optional[dict] = None,
        probability: Optional[float] = None,
        pool: Optional[dict] = None,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        isLogScale: Optional[bool] = None,
        resolutionTime: Optional[int] = None,
        resolution: Optional[str] = None,
        resolutionProbability: Optional[float] = None,
        lastUpdatedTime: Optional[int] = None,
        lastBetTime: Optional[int] = None,
        marketTier: Optional[str] = None,
        siblingContractId: Optional[str] = None,
        slug: Optional[str] = None,
        token: Optional[str] = None,
        lastCommentTime: Optional[int] = None,
        resolverId: Optional[str] = None,
        answers: Optional[list[str]] = None,
        shouldAnswersSumToOne: Optional[bool] = None,
        addAnswersMode: Optional[AddAnswersMode] = None,
        options: Optional[list[Option]] = None,
        totalBounty: Optional[float] = None,
        bountyLeft: Optional[float] = None,
        coverImageUrl: Optional[str] = None,
        groupSlugs: Optional[list[str]] = None,
    ):
        super().__init__(
            id,
            creatorId,
            creatorUsername,
            creatorName,
            createdTime,
            question,
            url,
            outcomeType,
            mechanism,
            probability,
            pool,
            totalLiquidity,
            volume,
            volume24Hours,
            isResolved,
            uniqueBettorCount,
            creatorAvatarUrl,
            closeTime,
            p,
            value,
            min,
            max,
            isLogScale,
            resolutionTime,
            resolution,
            resolutionProbability,
            lastUpdatedTime,
            lastBetTime,
            marketTier,
            siblingContractId,
            slug,
            token,
            lastCommentTime,
            resolverId,
        )
        self.description = description
        self.textDescription = textDescription
        self.answers = answers
        self.shouldAnswersSumToOne = shouldAnswersSumToOne
        self.addAnswersMode = addAnswersMode
        self.options = options
        self.totalBounty = totalBounty
        self.bountyLeft = bountyLeft
        self.coverImageUrl = coverImageUrl
        self.groupSlugs = groupSlugs

    def __repr__(self):
        return f"FullMarket(id={self.id}, question={self.question}, description={self.textDescription})"

    def __str__(self):
        return self.__repr__()


class Fees:
    def __init__(
        self,
        platformFee: float,
        liquidityFee: float,
        creatorFee: float,
    ):
        self.platformFee = platformFee
        self.liquidityFee = liquidityFee
        self.creatorFee = creatorFee


class Fill:
    def __init__(
        self,
        amount: int,
        matchedBetId: str,
        shares: float,
        timestamp: int,
    ):
        self.amount = amount
        self.matchedBetId = matchedBetId
        self.shares = shares
        self.timestamp = timestamp


class Bet:
    def __init__(
        self,
        shares: float,
        probBefore: float,
        isFilled: bool,
        probAfter: float,
        userId: str,
        amount: int,
        contractId: str,
        id: str,
        fees: Fees,
        isCancelled: bool,
        loanAmount: int,
        orderAmount: int,
        fills: list[Fill],
        createdTime: int,
        outcome: str,
    ):
        self.shares = shares
        self.probBefore = probBefore
        self.isFilled = isFilled
        self.probAfter = probAfter
        self.userId = userId
        self.amount = amount
        self.contractId = contractId
        self.id = id
        self.fees = fees
        self.isCancelled = isCancelled
        self.loanAmount = loanAmount
        self.orderAmount = orderAmount
        self.fills = fills
        self.createdTime = createdTime
        self.outcome = outcome
